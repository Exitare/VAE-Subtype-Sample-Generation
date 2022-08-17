import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['kl_loss'],label="kl_loss")
plt.title(
    # file.index.name+
          'Dense layer VAE embedding loss, latent dim = '+
         str(latent_dim))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.annotate('VAE layer type: Dense\ndtype: ?, normalized',
            xy=(.4, .8), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            # fontsize=20
            )
plt.legend(loc="lower left")
plt.savefig('rvrs_out/'+
            # file.index.name+'_'+
            str(500)+'_epochs__latent_dim_'+str(latent_dim)+'_2022-08-17_v4.png')