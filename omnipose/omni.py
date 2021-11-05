def omni_loss(self, lbl, y):
    veci = self._to_device(lbl[:,2:4]) #scaled to 5 in augmentation 
            dist = lbl[:,1] # now distance transform replaces probability
            boundary =  lbl[:,5]
            cellmask = dist>0
            w =  self._to_device(lbl[:,7])  # new smooth, boundary-emphasized weight calculated with augmentations  
            dist = self._to_device(dist)
            boundary = self._to_device(boundary)
            cellmask = self._to_device(cellmask).bool()
            flow = y[:,:2] # 0,1
            dt = y[:,2]
            bd = y[:,3]
            a = 10.
         
            wt = torch.stack((w,w),dim=1)
            ct = torch.stack((cellmask,cellmask),dim=1) 
            
            loss1 = 10.*self.criterion12(flow,veci,wt)  #weighted MSE 
            loss2 = self.criterion14(flow,veci,w,cellmask) #ArcCosDotLoss
            loss3 = self.criterion11(flow,veci,wt,ct)/a # DerivativeLoss
            loss4 = 2.*self.criterion2(bd,boundary)
            loss5 = 2.*self.criterion15(flow,veci,w,cellmask) # loss on norm 
            loss6 = 2.*self.criterion12(dt,dist,w) #weighted MSE 
            loss7 = self.criterion11(dt.unsqueeze(1),dist.unsqueeze(1),w.unsqueeze(1),cellmask.unsqueeze(1))/a  

    return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

# def omni_augment():
    
#     return 