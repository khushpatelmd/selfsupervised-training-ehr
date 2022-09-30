# @Author: Khush Patel, Zhi lab, drpatelkhush@gmail.com

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
    print("Remaining epochs are", remainingepochs)
    
    best_loss = np.inf

    mbdataset = MBDataset(rawdata, seed_value_changer=run_no_mask)
    loader = torch.utils.data.DataLoader(
    mbdataset, batch_size=batch_size, shuffle=True, num_workers= 12, pin_memory=True)
    
    #Resuming the last interrupted epoch
    train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=last_epoch, writer=writer, seed_value_changer=run_no_mask, steps_completed=steps_completed)
    print(f"Training loss at epoch {last_epoch} is {train_loss}")

    if train_loss<best_loss:
        best_loss = train_loss
            
        torch.save(model.state_dict(), save_wts_loc)
        print(f"Lowest training loss found at epoch {last_epoch}. Saving the model weights")    
    
    #Continuing from the next epoch
    for epoch in range(last_epoch+1, num_of_epochs):
        train_loss = train_fn(loader, model, optimizer=optim, device=device, scheduler=scheduler, epoch=last_epoch, writer=writer, seed_value_changer=run_no_mask)
        print(f"Training loss at epoch {epoch} is {train_loss}")

        if train_loss<best_loss:
            best_loss = train_loss
            
            torch.save(model.state_dict(), save_wts_loc)
            print(f"Lowest training loss found at epoch {epoch}. Saving the model weights")    
    
    writer.flush()
    writer.close()        
    sys.exit()