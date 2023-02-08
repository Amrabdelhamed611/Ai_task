import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os ,shutil ,copy ,time ,gc  ,random , timm
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,f1_score
from tqdm.notebook import tqdm
import torch ,torchinfo
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import utils
from torchvision import datasets, transforms ,models
from torch.cuda.amp import GradScaler ,autocast
from IPython import display as ipdisplay

class ModeL(nn.Module):
    """
    custom model class to train and valdite model 
    also track history and get insights by ploting loss , ACC , Confusion Matrix 
    """
    def __init__(self, n_classes,imagenet_weights=True,device= 'cuda'):
        super(ModeL, self).__init__()
        self.name = 'efficientnetv2_b2 '
        self.pre_trained_w = imagenet_weights
        
        self._creat_Model(n_classes,imagenet_weights)
        
        self.history =pd.DataFrame(columns=["Training_Loss","Training_Accuracy","Validation_Loss","Validation_Accuracy","epoch_time"])
        self.pramters_number = self.model_Info()
        
    def _creat_Model(self,out_features , pretrained=True):
        self.model = timm.create_model('tf_efficientnetv2_b2', pretrained=pretrained)   
        in_f = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=in_f, out_features=out_features, bias=True),
            nn.LogSoftmax(dim=1) ) 
        self.model.to(device)
        
    
    def model_Info(self,summary =False ):
        prams=0
        for p in list(self.model.parameters()):
            prams += p.nelement()
        prams = np.round(prams/10**6,2)
        if summary:
            print(f'model parameters: {prams} M')
            print(torchinfo.summary(model)) if summary else ''
        return prams
    
    def Load_Weights(self,load_weights_path ):
        state = torch.load(load_weights_path)
        self.load_state_dict(state['state_dict'])

    @autocast()    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def _track_history(self,T_Loss,T_Acc,V_Loss,V_Acc,t_ep):
        s = pd.Series([T_Loss,T_Acc,V_Loss,V_Acc,t_ep] ,index= self.history.columns)
        self.history = self.history.append(s,ignore_index=True)
        
    def save_model(self,best_metric,current_metric,name,epoch):
        if current_metric >= best_metric:
            state = {'epoch': epoch,
                     'state_dict': self.state_dict()}
            torch.save(state, name+'.pt')
            best_metric = current_metric
            print(f"%s is Saved with metric :%.3f" % (str(name).replace("_"," "),best_metric ))
        return best_metric
    
    def train_one_epoch(self, loader_train, criterion, optimizer,scaler ,device= 'cuda'):
        # keep track of training loss
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        ###################
        # train the model #
        ###################
        self.model.train()
        for  (data, target) in tqdm(loader_train):
            # move tensors to GPU if CUDA is available
            data ,target = data.to(device), target.to(device )
            #clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # calculate the batch loss
            predictions = self.model(data)
            loss = criterion(predictions, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            optimizer.step()
            acc= (predictions.argmax(dim=1) == target).cpu().numpy().mean()
            # one steps
            epoch_loss +=loss.detach().item()
            epoch_accuracy += acc
            #del predictions ,data , target ,loss,acc
            #gc.collect()
            #torch.cuda.empty_cache()
        return  epoch_loss/len(loader_train), epoch_accuracy/len(loader_train)
        
    def _validate_one_epoch(self, loader, criterion, device= 'cuda'):
        # keep track of validation loss
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        self.model.eval()
        y_df = pd.DataFrame(columns= ["Target","predictions"])
        ######################
        # validate the model #
        ######################
        for data, target in loader:
            # move tensors to GPU if CUDA is available
            data ,target = data.to(device), target.to(device)

            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                
                predictions = self.model(data)
                # calculate the batch loss
                loss = criterion(predictions, target)
                # Calculate Accuracy
                acc = (predictions.argmax(dim=1) == target).cpu().numpy().mean()
                # update average validation loss and accuracy
                epoch_loss += loss.detach().item()
                epoch_accuracy += acc
                
            y_df = y_df.append(pd.DataFrame({"Target":target.cpu().numpy().reshape(-1) ,
                                            "predictions":predictions.argmax(dim=1).cpu().numpy().reshape(-1)  }),
                                                   ignore_index=True)
        
       # del predictions ,data , target ,loss,acc
       # gc.collect()
        #torch.cuda.empty_cache()
        return  epoch_loss/len(loader), epoch_accuracy/len(loader),y_df 
    
    def validate_one_epoch(self, loader, criterion, device= 'cuda'):
        loss,acc,y_df =self._validate_one_epoch(loader, criterion, device= device)
        f1= f1_score(y_df["Target"].tolist(), y_df["predictions"].tolist(), average='macro')
        return loss,acc , f1
    
    def plot_history(self):
        fig,ax=plt.subplots(1,2,figsize = (16,4))
        ax[0]= self._plot(ax[0], self.history.index+1,self.history['Training_Loss'],self.history['Validation_Loss'],'Loss' )
        ax[1]= self._plot(ax[1], self.history.index+1,self.history['Training_Accuracy'],self.history['Validation_Accuracy'],'Accuracy')
        #ax[1].set_ylim([.20, .80])
        plt.show()
        
    def _plot(self, ax, ep,trian,val,plot_name ):
        ax.plot(ep, trian,label = "Training",color = 'blue')
        ax.plot(ep, val ,label = "Valdtion",color = 'red')
        ax.set_title(f"{plot_name} per Epoch",font ={'weight' : 'bold'}) 
        ax.legend()
        ax.set_xlabel("Epochs",font ={'weight' : 'bold'})
        ax.set_ylabel(f"{plot_name}",font ={'weight' : 'bold'})
        return ax
    
    def model_data_monitor(self,loader, criterion, device= 'cuda' ): 

        loss, acc, y_df = self._validate_one_epoch( loader, criterion, device= device)
        y_df["eq"] = (y_df["Target" ] == y_df["predictions" ])
        mapper = {v:k for k,v in loader.dataset.class_to_idx.items()}
        y_df["Target" ] = y_df["Target" ].map(mapper) 
        y_df["predictions" ] = y_df["predictions"].map(mapper) 
        g= y_df.groupby(['Target','eq'])\
                .agg('size').div(y_df.shape[0])\
                .sort_index( ascending=[True,False]).unstack()

        plt.rc('font', **{'size'   : 13})
        print(f'Accuracy : {g.sum(0).values[1]} ,Error:{g.sum(0).values[0]} ')
        fig,ax=plt.subplots(1,2,figsize = (16,6))
        ConfusionMatrixDisplay(confusion_matrix(y_df["Target" ], y_df["predictions" ]  )
                              ,display_labels=list(loader.dataset.classes) ).plot(ax=ax[0], xticks_rotation= 45)
        ax[0].set_title("Confusion Matrix",font ={'weight' : 'bold'})
        ax[0].set_xlabel('Predictions',font ={'weight' : 'bold'})
        ax[0].set_ylabel('True ',font ={'weight' : 'bold'})
        sns.heatmap(g,annot=True,ax=ax[1])
        ax[1].tick_params(axis='y', rotation=0)
        ax[1].set_xlabel('Predictions',font ={'weight' : 'bold'})
        ax[1].set_ylabel('classes',font ={'weight' : 'bold'})
        ax[1].set_title("Error / Accuracy for each Class",font ={'weight' : 'bold'})
        plt.show()
        #del g, fig,ax, y_df
        torch.cuda.empty_cache()
        return fig,ax
    
    def train_n_ep(self,epochs,loader_train,loader_val,criterion,optimizer,device = 'cuda'):
        best_metric = 0 if len(self.history)==0 else self.history["Validation_Accuracy"].max()
        scaler = GradScaler(enabled=True,)
        

        for epoch in range(1,epochs+1):
            start_time = time.time()
            loss, acc= self.train_one_epoch(loader_train, criterion, optimizer, scaler =scaler ,device= device)
            loss_val, acc_val ,f1_val = self.validate_one_epoch(loader_val, criterion, device)
            run_time= (time.time() - start_time)
            self._track_history(loss, acc,loss_val, acc_val,run_time)
            # ipdisplay.clear_output()
            print(f"[EPOCH]: %i, [LOSS]: %.6f, [ACC]: %.3f" % (epoch, loss, acc),
                  "\n[Val_LOSS]: %.6f, [Val_ACC]: %.3f, [Val_F1]: %.3f" % ( loss_val, acc_val , f1_val))

            best_metric= self.save_model(best_metric,f1_val,"Best_model_on_valdtion",epoch)
            
def load_image(path):
    img = io.read_image(path)
    IMG_SIZE = 260
    trans =transforms.Compose([ transforms.ToPILImage(),
                                    transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                            ])
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    img =img.to(device).type(torch.cuda.HalfTensor)
    return img


def predict(model):
    model.eval()
    classes =['bus', 'crossover', 'hatchback', 'motorcycle', 'pickup-truck', 'sedan', 'truck', 'van']
    out = model(image)
    idx = out.argmax().to('cpu').numpy()
    return classes[idx]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = load_image('test/bus/bus-front (16).jpg')
    pridection = predict(model  )
    print(pridection)