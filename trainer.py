import logging
import wandb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from writer import Writer
from utils import general_utils as utils


def id_loss_func(y_gt, y_pred):
    return tf.reduce_mean(tf.keras.losses.MAE(y_gt, y_pred))

def perc_model (vgg_model):
    output1 = vgg_model.layers[1].output
    output2 = vgg_model.layers[4].output
    output3 = vgg_model.layers[7].output
    output4 = vgg_model.layers[12].output
    output5 = vgg_model.layers[17].output
    perceptual_model = keras.Model(inputs=vgg_model.input,outputs=[output1,output2,output3,output4,output5])
    return perceptual_model


def perc_style_loss(image: tf.Tensor, output: tf.Tensor,perceptual_model: tf.keras.Model) -> tf.Tensor:
    image_v = keras.applications.vgg19.preprocess_input(image*255.0)
    output_v = keras.applications.vgg19.preprocess_input(output*255.0)

    output_f1, output_f2, output_f3, output_f4, output_f5 = perceptual_model(output_v) 
    image_f1, image_f2, image_f3, image_f4, image_f5 = perceptual_model(image_v)

    perc_f1 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f1-output_f1),axis=(1,2,3)))
    perc_f2 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f2-output_f2),axis=(1,2,3)))
    perc_f3 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f3-output_f3),axis=(1,2,3)))
    perc_f4 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f4-output_f4),axis=(1,2,3)))
    perc_f5 = tf.reduce_mean(tf.reduce_mean(tf.abs(image_f5-output_f5),axis=(1,2,3)))
    perceptual_loss = perc_f1 + perc_f2 + perc_f3 + perc_f4 + perc_f5
    # perceptual_loss /= 5 

    return perceptual_loss


class Trainer(object):
    def __init__(self, args, model, data_loader):
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)
        self.id_ll=[]
        self.mssim_ll=[]
        self.lnd_ll=[]
        self.l1_ll=[]
        self.total_ll=[]
        self.pixel_ll=[]
        self.perceptual_ll = []
          #WandB
	wandb.login(key='b56a5d7a531074d92b0cdc24f9e3df4c762267cc')
	wandb.init(project="model_with_Arcface_test_3_ahmad")
        self.vgg19_model = keras.applications.VGG19(include_top=False,input_shape=(256,256,3))
        self.perceptual_model = perc_model(self.vgg19_model)
        self.model = model
        self.data_loader = data_loader
        self.attr_test, self.id_test, self.mask_test, self.real_test, self.eye_img = self.data_loader.get_batch(is_cross=False)
        # lrs & optimizers
        lr = 5e-5 if self.args.resolution == 256 else 1e-5

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.g_gan_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * lr)
        self.w_d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.4 * lr)

        self.im_d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.4 * lr)

        # Losses
        self.gan_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.pixel_loss_func = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)

        self.id_loss_func = id_loss_func

        if args.pixel_mask_type == 'gaussian':
            sigma = int(80 * (self.args.resolution / 256))
            self.pixel_mask = utils.inverse_gaussian_image(self.args.resolution, sigma)
        else:
            self.pixel_mask = tf.ones([self.args.resolution, self.args.resolution])
            self.pixel_mask = self.pixel_mask / tf.reduce_sum(self.pixel_mask)

        self.pixel_mask = tf.broadcast_to(self.pixel_mask, [self.args.batch_size, *self.pixel_mask.shape])

        self.num_epoch = args.initial_epoch
        self.is_cross_epoch = False

        # Lambdas
        if args.unified:
            self.lambda_gan = 0.5
        else:
            self.lambda_gan = 1

        self.lambda_pixel = 1

        
        self.lambda_attr_id = 1
        self.lambda_landmarks = 0.001
        self.r1_gamma = 10
        
    
        self.lambda_id = 1
        self.lambda_l_w = 1
            

        # Test
        self.test_not_imporved = 0
        self.max_id_preserve = 0
        self.min_lnd_dist = np.inf

    def train(self):
        while self.num_epoch <= self.args.num_epochs:
            self.logger.info('---------------------------------------')
            self.logger.info(f'Start training epoch: {self.num_epoch}')

            if self.args.cross_frequency and (self.num_epoch % self.args.cross_frequency == 0):
                self.is_cross_epoch = True
#                 self.logger.info('This epoch is cross-face')
            else:
                self.is_cross_epoch = False
#                 self.logger.info('This epoch is same-face')

            try:

                if self.num_epoch % self.args.test_frequency == 0:
                    self.test()
                    
                self.train_epoch()
                

                

            except Exception as e:
                self.logger.exception(e)
                raise

            if self.test_not_imporved > self.args.not_improved_exit:
                self.logger.info(f'Test has not improved for {self.args.not_improved_exit} epochs. Exiting...')
                break

            self.num_epoch += 1

    def train_epoch(self):
        id_loss = 0
        landmarks_loss = 0
        g_w_gan_loss = 0
        pixel_loss = 0
        w_d_loss = 0
        w_loss = 0

        self.logger.info(f'train in epoch: {self.num_epoch}')
        self.model.train()

        use_w_d = self.args.W_D_loss

        # if use_w_d and use_im_d and not self.args.unified:
        if not self.args.unified:
            if self.num_epoch % 2 == 0:
                # This epoch is not using image_D
                use_im_d = False

            if self.num_epoch % 2 != 0:
                # This epoch is not using W_D
                use_w_d = False


        attr_img, id_img, id_mask, real_img, eye_img = self.data_loader.get_batch(is_cross=self.is_cross_epoch)

        id_embedding = self.model.G.id_encoder(eye_img)
        id_embedding_for_loss = self.model.G.pretrained_id_encoder(id_mask)
        src_landmarks = self.model.G.landmarks(id_img)  
        attr_input = id_mask
        
        with tf.GradientTape(persistent=True) as g_tape:

            attr_out = self.model.G.attr_encoder(attr_input)
            attr_embedding = attr_out

            z_tag = tf.concat([id_embedding, attr_embedding], -1)
            w = self.model.G.latent_spaces_mapping(z_tag)
            pred = self.model.G.stylegan_s(w)

            # Move to roughly [0,1]
            pred = (pred + 1) / 2

            if use_w_d:
                with tf.GradientTape() as w_d_tape:
                    fake_w_logit = self.model.W_D(fake_w)
                    g_w_gan_loss = self.generator_gan_loss(fake_w_logit)

                    with g_tape.stop_recording():
                        real_w_logit = self.model.W_D(real_w)
                        w_d_loss = self.discriminator_loss(fake_w_logit, real_w_logit)
                        w_d_total_loss = w_d_loss

                        if self.args.gp:
                            w_d_gp = self.R1_gp(self.model.W_D, real_w)
                            w_d_total_loss += w_d_gp
       

            if self.args.id_loss:
                pred_id_embedding = self.model.G.pretrained_id_encoder(pred)
                id_loss = self.lambda_id * id_loss_func(pred_id_embedding, tf.stop_gradient(id_embedding_for_loss))
                self.id_ll.append(id_loss)
            if self.args.landmarks_loss:
                try:
                    dst_landmarks = self.model.G.landmarks(pred)
                except Exception as e:
                    self.logger.warning(f'Failed finding landmarks on prediction. Dont use landmarks loss. Error:{e}')
                    dst_landmarks = None

                if dst_landmarks is None or src_landmarks is None:
                    landmarks_loss = 0
                else:
                    landmarks_loss = self.lambda_landmarks * \
                                     tf.reduce_mean(tf.keras.losses.MSE(src_landmarks, dst_landmarks))
                self.lnd_ll.append(landmarks_loss)

            
            
            perceptual_loss = 0.01 * perc_style_loss(id_img,pred,self.perceptual_model)
            self.perceptual_ll.append(perceptual_loss)

            if not self.is_cross_epoch and self.args.pixel_loss:
                l1_loss = self.pixel_loss_func(id_img, pred, sample_weight=self.pixel_mask)

                self.l1_ll.append(l1_loss)
                if self.args.pixel_loss_type == 'mix':
                    mssim = tf.reduce_mean(1 - tf.image.ssim(id_img, pred, 1.0))
                    self.mssim_ll.append(mssim)

                    pixel_loss = self.lambda_pixel * (0.84 * mssim + 0.16 * l1_loss)
                else:
                    pixel_loss = self.lambda_pixel * l1_loss
                    
                self.pixel_ll.append(pixel_loss)


            g_gan_loss = g_w_gan_loss

            total_g_not_gan_loss = id_loss \
                                   + landmarks_loss \
                                   + pixel_loss \
                                   + perceptual_loss
            self.total_ll.append(total_g_not_gan_loss)
            self.logger.info(f'total G (not gan) loss is {total_g_not_gan_loss:.3f}')
            self.logger.info(f'G gan loss is {g_gan_loss:.3f}')

            
        if self.num_epoch%100==0:
            wandb.log({"epoch": self.num_epoch, "id_loss": np.mean(self.id_ll),"Lnd_loss": np.mean(self.lnd_ll),
            "l1_loss": np.mean(self.l1_ll),"pixel_loss":np.mean(self.pixel_ll),"perceptual_loss":np.mean(self.perceptual_ll),
            "total_g_not_gan_loss":np.mean(self.total_ll),"g_w_gan_loss":g_w_gan_loss,
             "gt_img": wandb.Image(id_img[0]) ,  "mask_img": wandb.Image(id_mask[0]) ,  "pred_img": wandb.Image(pred[0]) ,  "eye_img": wandb.Image(eye_img[0])})
            

            self.id_ll=[]
            self.lnd_ll=[]
            self.l1_ll=[]
            self.pixel_ll=[]
            self.total_ll=[]
            self.perceptual_ll=[]

        
        if total_g_not_gan_loss != 0:
            g_grads = g_tape.gradient(total_g_not_gan_loss, self.model.G.trainable_variables)

            g_grads_global_norm = tf.linalg.global_norm(g_grads)
            self.logger.info(f'global norm G not gan grad: {g_grads_global_norm}')

            self.g_optimizer.apply_gradients(zip(g_grads, self.model.G.trainable_variables))

        if use_w_d:
            g_gan_grads = g_tape.gradient(g_gan_loss, self.model.G.trainable_variables)

            g_gan_grad_global_norm = tf.linalg.global_norm(g_gan_grads)
            self.logger.info(f'global norm G gan grad: {g_gan_grad_global_norm}')

            self.g_gan_optimizer.apply_gradients(zip(g_gan_grads, self.model.G.trainable_variables))

            w_d_grads = w_d_tape.gradient(w_d_total_loss, self.model.W_D.trainable_variables)
            self.w_d_optimizer.apply_gradients(zip(w_d_grads, self.model.W_D.trainable_variables))

        del g_tape

    # Common

    # Test
    def test(self):
        self.model.my_save(f'_my_save_epoch_{self.num_epoch}')





    def test_reconstruction(self, img, errors_dict, display=False, display_name=None):
        pred, id_embedding, w, attr_embedding, src_lnds = self.model.G(img, img, img)

        recon_image = tf.clip_by_value(pred, 0, 1)
        recon_pred_id = self.model.G.id_encoder(recon_image)

        mse = tf.reduce_mean((img - recon_image) ** 2, axis=[1, 2, 3]).numpy()
        psnr = tf.image.psnr(img, recon_image, 1).numpy()

        errors_dict['MSE'].extend(mse)
        errors_dict['PSNR'].extend(psnr)
        errors_dict['ID'].extend(tf.keras.losses.cosine_similarity(id_embedding, recon_pred_id).numpy())

        if display:
            Writer.add_image(f'reconstruction/{display_name}', pred, step=self.num_epoch)

    # Helpers

    def generator_gan_loss(self, fake_logit):
        """
        G logistic non saturating loss, to be minimized
        """
        g_gan_loss = self.gan_loss_func(tf.ones_like(fake_logit), fake_logit)
        return self.lambda_gan * g_gan_loss

    def discriminator_loss(self, fake_logit, real_logit):
        """
        D logistic loss, to be minimized
        verified as identical to StyleGAN's loss.D_logistic
        """
        fake_gt = tf.zeros_like(fake_logit)
        real_gt = tf.ones_like(real_logit)

        d_fake_loss = self.gan_loss_func(fake_gt, fake_logit)
        d_real_loss = self.gan_loss_func(real_gt, real_logit)

        d_loss = d_real_loss + d_fake_loss

        return self.lambda_gan * d_loss

    def R1_gp(self, D, x):
        with tf.GradientTape() as t:
            t.watch(x)
            pred = D(x)
            pred_sum = tf.reduce_sum(pred)

        grad = t.gradient(pred_sum, x)

        # Reshape as a vector
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean(norm ** 2)
        gp = 0.5 * self.r1_gamma * gp

        return gp
