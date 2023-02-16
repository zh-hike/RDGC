

def train_epoch(engine):

    for batch, (inputs, targets, miss_matrixs) in enumerate(engine.dataloader):
        inputs = [x.cuda() for x in inputs]
        targets = targets.cuda()
        miss_matrixs = [miss.cuda() for miss in miss_matrixs]
        out, fusion2recon, latents = engine.model(inputs, miss_matrixs)
        output = {'preds': out, 
                  'reals': inputs, 
                  'fusion2recon': fusion2recon, 
                  'miss_matrixs': miss_matrixs,
                  'latents': latents}
        loss_dict = engine.loss_func(output)
        
        engine.optimizer.zero_grad()
        loss = loss_dict['loss']
        loss.backward()
        engine.optimizer.step()
        print(loss.item())