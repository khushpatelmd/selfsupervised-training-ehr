# @Author: Khush Patel, Zhi lab

from config import *
from model import *


def train_fn(data_loader, model, optimizer, device, scheduler, epoch, writer, seed_value_changer, steps_completed=0):
    model = model.train()
    final_loss = 0
    final_acc =  []
    running_loss = 0
    loop = tqdm(data_loader, leave=True)  #If leave =True, keeps all traces of the progressbar upon termination of iteration
    counter = 0
    running_acc = 0
    scaler = torch.cuda.amp.GradScaler()
    
    for batch in loop:

        optimizer.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        segment_type_ids = batch['segment_type_ids'].to(device)
        masked_indices = batch['masked_indices'].to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels, token_type_ids=segment_type_ids)
        # extract loss
            loss = outputs.loss
            
        # backpropogation
        scaler.scale(loss).backward()
        # update parameters
        scaler.step(optimizer)
        # scheduler
        scheduler.step()
        scaler.update()

        running_loss += loss.item()

        predictions = outputs.logits
        preds= (predictions.argmax(dim=2)).cpu().numpy()
        labels= labels.cpu().numpy()
        masked_indices = masked_indices.cpu().numpy()
    
        acc = []
        for i in range(predictions.shape[0]):
            
            mask = masked_indices[i]
            pred = preds[i][mask]
            label = labels[i][mask]

            if len(label)!=0:
                acc.append(accuracy_score(label, pred))
            
        running_acc += mean(acc)      
        
        
        # saving checkpoint
        if (counter % save_every_step == 0) & (counter!=0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / save_every_step,
                'number_of_training_steps': counter,
                'scheduler': scheduler,
                "seed_value_changer": seed_value_changer
            }, save_dir, pickle_module=dill)
            
        if (counter % measure_metrics_steps == 0) & (counter!=0):
            
            writer.add_scalar('MLM accuracy', running_acc/measure_metrics_steps, steps_completed + seed_value_changer * len(data_loader) * num_of_epochs  +  epoch * len(data_loader) + counter)
            writer.add_scalar('Training loss', running_loss/measure_metrics_steps, steps_completed + seed_value_changer * len(data_loader) * num_of_epochs  +  epoch * len(data_loader) + counter)
            
            
            logging.info(
                    f"The value of mean MLM accuracy for steps {steps_completed + seed_value_changer * len(data_loader) * num_of_epochs  +  epoch * len(data_loader) + counter} was {running_acc/measure_metrics_steps}")
            logging.info(
                    f"The value of loss for steps {steps_completed + seed_value_changer * len(data_loader) * num_of_epochs  +  epoch * len(data_loader) + counter} was {running_loss/measure_metrics_steps}")
            running_loss = 0
            running_acc = 0
            
        # Updating step counter
        counter += 1
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item(), MLM_accuracy=mean(acc))
        final_loss += loss.item()
    
    
    return  final_loss/counter

