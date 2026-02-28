from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator

def build_gan(z_dim=100):
    return Generator(z_dim), Discriminator()
