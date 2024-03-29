import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import keras
from keras.preprocessing.image import ImageDataGenerator
matplotlib.use('agg')
PLOT_PATH = 'cnn/plots/'
# imgs,titles = next(train_batches)
# plot(imgs,titles=labels)
# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("cnn/plots/confusion_matrix.jpg")
         

def get_img_fit_flow(image_config, fit_smpl_size, directory, target_size, batch_size, shuffle):   
    '''                                                                            
    Sample the generators to get fit data    
    image_config  dict   holds the vars for data augmentation & 
    fit_smpl_size float  subunit multiplier to get the sample size for normalization
    
    directory     str    folder of the images
    target_size   tuple  images processed size
    batch_size    str    
    shuffle       bool
    '''                                                                            
    if 'featurewise_std_normalization' in image_config and image_config['image_config']:                                      
       img_gen = ImageDataGenerator()                                              
       batches = img_gen.flow_from_directory(                                      
          directory=directory,                                                     
          target_size=target_size,                                                 
          batch_size=batch_size,                                                   
          shuffle=shuffle,                                                         
        )                                                                          
       fit_samples = np.array([])                                                  
       fit_samples.resize((0, target_size[0], target_size[1], 3))                  
       for i in range(batches.samples/batch_size):                                 
           imgs, labels = next(batches)                                            
           idx = np.random.choice(imgs.shape[0], batch_size*fit_smpl_size, replace=False)     
           np.vstack((fit_samples, imgs[idx]))                                     
    new_img_gen = ImageDataGenerator(**image_config)                               
    if 'featurewise_std_normalization' in image_config and image_config['image_config']:                                      
        new_img_gen.fit(fit_samples)                                               
    return new_img_gen.flow_from_directory(                                        
       directory=directory,                                                        
       target_size=target_size,                                                    
       batch_size=batch_size,                                                      
       shuffle=shuffle,                                                            
    )


class TrainingPlot(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(PLOT_PATH+'Epoch-{}.png'.format(epoch))
            plt.close()










# import tensorboard
# from keras.callbacks import TensorBoard
# https://medium.com/@kapilvarshney/how-to-plot-the-model-training-in-keras-using-custom-callback-function-and-using-tensorboard-41e4ce3cb401
