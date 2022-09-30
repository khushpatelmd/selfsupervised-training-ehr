# @Author: Khush Patel, Zhi lab
#The idea is to change the seed of mask (run_no_mask) every few epochs. This is used for resuming training from last checkpoint saved.

# local imports
import model as m
from config import *
from dataset_MLM import *
from engine_amp_MLM import *

if __name__ == "__main__":
    writer = SummaryWriter(tbpath)
    with open(raw_data_path, "rb") as f:
        rawdata = pickle.load(f)
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    
    model = m.model
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
    optim, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    checkpoint = torch.load(save_dir, pickle_module=dill)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    scheduler = checkpoint['scheduler']
    steps_completed = checkpoint['number_of_training_steps']

    remainingepochs = num_of_epochs - last_epoch 
    print(remainingepochs)
    
    best_loss = np.inf
    
    #Commenting out already completed epochs
    
    # for epoch in range(num_of_epochs):
    #     train_loss= train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer,seed_value_changer=run_no_mask, steps_completed=steps_completed)
    #     print(f"Training loss at epoch {epoch} is {train_loss}")

    #     if train_loss<best_loss:
    #         best_loss = train_loss
            
    #         torch.save(model.state_dict(), save_wts_loc)
    #         print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")
    
    # ####Changing mask seed
    # print("Changing the mask")
    # mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+1)
    # loader = torch.utils.data.DataLoader(
    # mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    # for epoch in range(num_of_epochs):
    #     train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+1, steps_completed=steps_completed)
    #     print(f"Training loss at epoch {epoch} is {train_loss}")

    #     if train_loss<best_loss:
    #         best_loss = train_loss
            
    #         torch.save(model.state_dict(), save_wts_loc)
    #         print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")    
    
    # ####Changing mask seed
    # print("Changing the mask")
    # mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+2)
    # loader = torch.utils.data.DataLoader(
    # mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    # for epoch in range(num_of_epochs):
    #     train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+2, steps_completed=steps_completed)
    #     print(f"Training loss at epoch {epoch} is {train_loss}")
        
    #     if train_loss<best_loss:
    #         best_loss = train_loss
            
    #         torch.save(model.state_dict(), save_wts_loc)
    #         print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")    

    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+3)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(last_epoch+1, num_of_epochs):
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+3, steps_completed=steps_completed)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")    
    

    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+4)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+4, steps_completed=steps_completed)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")       
            
    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+5)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+5, steps_completed=steps_completed)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights") 
            

            
    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+6)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss= train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+6, steps_completed=steps_completed)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights") 

    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+7)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+7, steps_completed=steps_completed)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights") 
            
    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+8)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+8, steps_completed=steps_completed)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")              
            
    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+9)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss= train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+9, steps_completed=steps_completed)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")      
    
    
    writer.flush()
    writer.close()        
    sys.exit()