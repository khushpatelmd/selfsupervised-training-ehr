# @Author: Khush Patel, Zhi lab

# Local imports
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
    
    best_loss = np.inf
    for epoch in range(num_of_epochs):
        train_loss= train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask)
        print(f"Training loss at epoch {epoch} is {train_loss}")
        
        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")
    
    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+1)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+1)
        print(f"Training loss at epoch {epoch} is {train_loss}")
        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")    
    
    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+2)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss= train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+2)
        print(f"Training loss at epoch {epoch} is {train_loss}")
        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")    

    ####Changing mask seed
    print("Changing the mask")
    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask+3)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    for epoch in range(num_of_epochs):
        train_loss= train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+3)
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
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+4)
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
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+5)
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
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+6)
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
        train_loss= train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+7)
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
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+8)
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
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=epoch, writer=writer, seed_value_changer=run_no_mask+9)
        print(f"Training loss at epoch {epoch} is {train_loss}")
        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")      
    
    
    writer.flush()
    writer.close()        
    sys.exit()