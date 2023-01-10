from pathlib import Path

from tqdm import tqdm
import tensorflow as tf
from model import landmarks
from writer import Writer
from utils import general_utils as utils
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib.image import imread 


class Inference(object):
    def __init__(self, args, model):
        self.args = args
        self.G = model.G
        self.pixel_loss_func = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)

    def infer_pairs(self):
        names = [f for f in self.args.id_dir.iterdir() if f.suffix[1:] in self.args.img_suffixes]
        for img_name in tqdm(names):
            id_path = utils.find_file_by_str(self.args.id_dir, img_name.stem)
            mask_path = utils.find_file_by_str(self.args.mask_dir, img_name.stem)
            attr_path = utils.find_file_by_str(self.args.attr_dir, img_name.stem)
            if len(id_path) != 1 or len(attr_path) != 1:
                print(f'Could not find a single pair with name: {img_name.stem}')
                continue

            id_img = utils.read_image(id_path[0], self.args.resolution)
            mask_img , attr_img = utils.read_mask_image(id_path[0], mask_path[0], self.args.resolution)


            out_img = self.G(mask_img, attr_img, id_img)[0]
			
            pred_id_embedding = self.G.id_encoder(out_img)
            gt_id_embedding = self.G.id_encoder(id_img)
            id_loss = tf.reduce_mean(tf.keras.losses.MAE(gt_id_embedding, pred_id_embedding))
            lnd_loss = tf.constant(0)  
            L1_loss  = self.pixel_loss_func(id_img, out_img)
            
            utils.save_image(out_img, self.args.output_dir.joinpath(f'{img_name.name}'))

    def opt_infer_pairs(self):
        names = [f for f in self.args.id_dir.iterdir() if f.suffix[1:] in self.args.img_suffixes]
        # names.extend([f for f in self.args.attr_dir.iterdir() if f.suffix[1:] in self.args.img_suffixes])
        for img_name in tqdm(names):
            id_path = utils.find_file_by_str(self.args.id_dir, img_name.stem)
            mask_path = utils.find_file_by_str(self.args.mask_dir, img_name.stem)
            eye_path = utils.find_file_by_str(self.args.eye_dir, img_name.stem)
            attr_path = utils.find_file_by_str(self.args.attr_dir, img_name.stem)
            if len(id_path) != 1 or len(attr_path) != 1:
                print(f'Could not find a single pair with name: {img_name.stem}')
                continue

            id_img = utils.read_image(id_path[0], self.args.resolution)
            gt_img = id_img
            mask_img , attr_img = utils.read_mask_image(id_path[0], mask_path[0], self.args.resolution)
            eye_img = utils.read_eye_image( eye_path[0], self.args.resolution)
		
            out_img = self.G(eye_img, attr_img, id_img)[0]
            id_embedding = self.G(eye_img, attr_img, id_img)[1]
            attr_embedding = self.G(eye_img, attr_img, id_img)[2]			
            z_tag = tf.concat([id_embedding, attr_embedding], -1)
            w = self.G.latent_spaces_mapping(z_tag)
            pred = self.G.stylegan_s(w)
            pred = (pred + 1) / 2
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1 =0.9, beta_2=0.999, epsilon=1e-8 ,name='Adam')
            loss =  tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
            mask = Image.open(mask_path[0])
            mask = mask.convert('RGB')
            mask = mask.resize((256,256))
            mask = np.asarray(mask).astype(float)/255.0
            mask1 = np.asarray(mask).astype(float)  
            # mask1 = np.expand_dims(mask1 , axis=0) 
            loss_value = 0
            wp = tf.Variable(w ,trainable=True)
            for i in range(5000):
                print('iteration:{0}   loss value is: {1}'.format(i,loss_value))
                with tf.GradientTape() as tape:
                    out_img = self.G.stylegan_s(wp) 
                    out_img = (out_img + 1)  / 2 
                    # utils.save_image(out_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_out.png'))                        
                    mask_out_img = out_img * mask1
                    if i%200==0:
                        utils.save_image(out_img , self.args.output_dir.joinpath(f'{img_name.name[:-4]}' + '_out_{0}.png'.format(i)))
                    # utils.save_image(mask_out_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_test.png'))
                    # utils.save_image(mask_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_m.png'))
                    loss_value = loss(mask_img ,mask_out_img)
                                        
                grads = tape.gradient(loss_value, [wp])
                optimizer.apply_gradients(zip(grads, [wp]))
                
            
            opt_pred = self.G.stylegan_s(wp)
            opt_pred = (opt_pred + 1) / 2

            utils.save_image(pred, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_init.png'))
            utils.save_image(opt_pred, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_final.png'))
            utils.save_image(id_img, self.args.output_dir.joinpath(f'{img_name.name[:-4]}'+'_gt.png'))
		
    def infer_on_dirs(self):
        attr_paths = list(self.args.attr_dir.iterdir())
        attr_paths.sort()

        id_paths = list(self.args.id_dir.iterdir())
        id_paths.sort()

        for attr_num, attr_img_path in tqdm(enumerate(attr_paths)):
            if not attr_img_path.is_file() or attr_img_path.suffix[1:] not in self.args.img_suffixes:
                continue

            attr_img = utils.read_image(attr_img_path, self.args.resolution, self.args.reals)

            attr_dir = self.args.output_dir.joinpath(f'attr_{attr_num}')
            attr_dir.mkdir(exist_ok=True)

            utils.save_image(attr_img, attr_dir.joinpath(f'attr_image.png'))

            for id_num, id_img_path in enumerate(id_paths):
                if not id_img_path.is_file() or id_img_path.suffix[1:] not in self.args.img_suffixes:
                    continue

                id_img = utils.read_image(id_img_path, self.args.resolution, self.args.reals)

                pred = self.G(id_img, attr_img)[0]

                utils.save_image(pred, attr_dir.joinpath(f'prediction_{id_num}.png'))
                utils.save_image(id_img, attr_dir.joinpath(f'id_{id_num}.png'))

    def interpolate(self, w_space=True):
        # Change to 0,1 for interpolation
        extra_start = 0
        extra_end = 1
        L = extra_end - extra_start
        # Extrapolation values include the 0,1 iff
        #   N-1 is divisible by L if including endpoint
        #   N is divisble by L o.w
        #   where L is the length of the extrapolation range ( L = b-a for [a,b] )
        #   and N is number of jumps
        num_jumps = 8 * L + 1

        for d in self.args.input_dir.iterdir():
            out_d = self.args.output_dir.joinpath(d.name)
            out_d.mkdir(exist_ok=True)

            ids = list(d.glob('*id*'))
            attrs = list(d.glob('*attr*'))

            if len(ids) == 1 and len(attrs) == 2:
                const = 'id'
            elif len(ids) == 2 and len(attrs) == 1:
                const = 'attr'
            else:
                print(f'Wrong data format for {d.name}')
                continue

            if const == 'id':
                start_img = utils.read_image(attrs[0], self.args.resolution, self.args.real_attr)
                end_img = utils.read_image(attrs[1], self.args.resolution, self.args.real_attr)
                const_img = utils.read_image(ids[0], self.args.resolution, self.args.real_id)

                if self.args.loop_fake:
                    if not self.args.real_attr:
                        start_img = self.G(start_img, start_img)
                        end_img = self.G(end_img, end_img)
                    if not self.args.real_id:
                        const_img = self.G(const_img, const_img)

                const_id = self.G.id_encoder(const_img)
                start_attr = self.G.attr_encoder(start_img)
                end_attr = self.G.attr_encoder(end_img)

                s_z = tf.concat([const_id, start_attr], -1)
                e_z = tf.concat([const_id, end_attr], -1)

            elif const == 'attr':
                start_img = utils.read_image(ids[0], self.args.resolution, self.args.real_id)
                end_img = utils.read_image(ids[1], self.args.resolution, self.args.real_id)
                const_img = utils.read_image(attrs[0], self.args.resolution, self.args.real_attr)

                if self.args.loop_fake:
                    if not self.args.real_attr:
                        const_img = self.G(const_img, const_img)[0]
                    if not self.args.real_id:
                        start_img = self.G(start_img, start_img)[0]
                        end_img = self.G(end_img, end_img)[0]

                start_id = self.G.id_encoder(start_img)
                end_id = self.G.id_encoder(end_img)

                const_attr = self.G.attr_encoder(const_img)

                s_z = tf.concat([start_id, const_attr], -1)
                e_z = tf.concat([end_id, const_attr], -1)


            utils.save_image(const_img, out_d.joinpath(f'const_{const}.png'))
            utils.save_image(start_img, out_d.joinpath(f'start.png'))
            utils.save_image(end_img, out_d.joinpath(f'end.png'))

            if w_space:
                s_w = self.G.latent_spaces_mapping(s_z)
                e_w = self.G.latent_spaces_mapping(e_z)
                for i in range(num_jumps):
                    inter_w = (1 - i / num_jumps) * s_w + (i / num_jumps) * e_w
                    out = self.G.stylegan_s(inter_w)
                    out = (out + 1) / 2
                    utils.save_image(out[0],
                                     out_d.joinpath(f'inter_{i:03}.png'))
            else:
                for i in range(num_jumps):
                    inter_z = (1 - i / num_jumps) * s_z + (i / num_jumps) * e_z
                    inter_w = self.G.latent_spaces_mapping(inter_z)
                    out = self.G.stylegan_s(inter_w)
                    out = (out + 1) / 2
                    utils.save_image(out[0],
                                     out_d.joinpath(f'inter_{i:03}.png'))

