from prediction import prediction
import matplotlib.pyplot as plt
import fire 

def predictFromTerminal(image_path):
    annotatedImage = prediction(image_path)
    plt.imshow(annotatedImage)
    plt.grid(False)
    plt.axis('off')
    plt.show()



if __name__=='__main__':
    print("Starting execution:")
    fire.Fire(predictFromTerminal)
