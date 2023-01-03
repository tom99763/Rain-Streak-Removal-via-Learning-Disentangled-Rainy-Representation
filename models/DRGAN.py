import sys
sys.path.append('./models')
import tensorflow as tf
from modules import *
from losses import *
from discriminator import *


class DRGAN(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    
    self.G = Generator(config)
    self.Dr = Discriminator(config)
    self.Dnr = Discriminator(config)
    self.config = config
    
  def train_step(self, inputs):
    x_nr, x_r = inputs
    
    with tf.GradientTape(persistent=True) as tape:
      #no rain
      c_nr, f_nr = self.G.encode(x_nr)
      
      #rain generation
      z = tf.random.normal((x_nr.shape[0], self.config['latnet_dim']))
      x_nr2r = self.G.decode(c_nr, z)
      c_nr2r, f_nr2r = self.G.encode(x_nr2r)
      
      #deraining
      c_r, f_r = self.G.encode(x_r)
      x_r2nr = self.G.decode(c_r, tf.zeros_like(f_r))
      c_r2nr, f_r2nr = self.encode(x_r2nr)
      
      #reconstruction
      x_nr_re = self.G.decode(c_nr, tf.zeros_like(f_nr))
      x_r_re = self.G.decode(c_r, f_r)
      
      #discirmination
      critic_real_r = self.Dr(x_nr2r)
      critic_fake_r = self.Dr(x_r)
      critic_real_nr = self.Dnr(x_r2nr)
      critic_fake_nr = self.Dnr(x_nr)
      
      ### compute losses
      #latent reconstruction
      l_central = l1_loss(f_nr, 0.) + l1_loss(f_r2nr, 0.)
      l_match = l1_loss(z, x_nr2r)
      l_latent = l_central + l_match
      
      #reconstruction
      l_re = l1_loss(x_nr_re, x_nr) + l1_loss(x_r_re, x_r)
      
      #adversarial 
      l_g_r, l_d_r = gan_loss(critic_real_r, critic_fake_r, self.config['gan_type'])
      l_g_nr, l_d_nr = gan_loss(critic_real_nr, critic_fake_nr, self.config['gan_type'])
      l_g = l_g_r + l_g_nr
      l_d = l_d_r + l_d_nr
      
      #total loss
      g_loss = l_re + l_latent + l_g
      d_loss = l_d
    g_grads = tape.gradient(g_loss, self.G.trainable_weights)
    d_grads = tape.gradient(d_loss, self.Dr.trainable_weights + self.Dnr.trainable_weights)
    self.optimizer[0].apply_gradients(zip(g_grads, self.G.trainable_weights))
    self.optimizer[1].apply_gradients(zip(d_grads, self.Dr.trainable_weights + self.Dr.trainable_weights))
    return {'l_central':l_central, 'l_match':l_match, 'l_re':l_re, 'l_g':l_g, 'l_d':l_d}
    
    
    
    
    
      
      
