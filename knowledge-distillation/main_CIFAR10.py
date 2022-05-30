import torch
import torch.nn.functional as F
from tqdm import tqdm

from resnet import resnet20, resnet44   
from dataset import Dataset

FLAGS = {
    "USE_PRETRAINED": True   
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # cpu on my laptop
# Training hyperparameters
OPTIMIZER_FUNC = torch.optim.AdamW
SCHEDULER_FUNC = torch.optim.lr_scheduler.MultiStepLR
NUM_EPOCHS = 200
NUM_EPOCHS_STOP = 200
MAX_LEARNING_RATE = 0.1
MOMENTUM = 0.9
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
GAMMA = 0.1 # Default
# Knowledge Distillation settings
NUM_STUDENTS = 3
TEMPERATURE = 4
LAMBDA = 0.1

# Get dataloaders
dataset = Dataset()
train_loader, test_loader, classes = dataset.get_CIFAR10()

# STAGE 1: TRAIN TEACHER
# Init teacher
skip_training_teacher = False
if not skip_training_teacher:
    def train(model, train_loader):
        model.train()

        epoch_loss = torch.zeros([1]).to(DEVICE)
        corrects   = torch.zeros([1]).to(DEVICE)

        trainloader_tqdm = tqdm(train_loader, unit = 'batch', leave = False)
        trainloader_tqdm.set_description_str('Training teacher')
        for (inputs, labels) in trainloader_tqdm:
            # Load data and move to cuda
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            label_onehots = F.one_hot(labels, num_classes = 10).float()
            # forward
            outputs = model(inputs)
            batch_loss = F.kl_div(F.log_softmax(outputs, dim = 1),
                                  label_onehots,
                                  reduction = 'sum')
            # backward
            teacher.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), GRAD_CLIP)
            teacher.optimizer.step()

            # store batch results
            epoch_loss += batch_loss * inputs.size(0)
            predictions = torch.max(outputs, dim = 1)[1]
            corrects += torch.sum(predictions == labels)
            trainloader_tqdm.set_postfix({'epoch_loss': epoch_loss.item(),
                                          'corrects':   corrects.item()})

        teacher.metrics['train_loss'] = torch.cat([teacher.metrics['train_loss'], epoch_loss/train_loader.dataset.__len__()])
        teacher.metrics['train_accuracy'] = torch.cat([teacher.metrics['train_accuracy'], corrects/train_loader.dataset.__len__()])

        return teacher

    def test(model, test_loader, flag_baseline = False):
        model.eval()

        epoch_loss = torch.zeros([1]).to(DEVICE)
        corrects   = torch.zeros([1]).to(DEVICE)

        testloader_tqdm = tqdm(test_loader, unit = 'batch', leave = False)
        testloader_tqdm.set_description_str('Testing teacher')
        with torch.no_grad():
            for (inputs, labels) in testloader_tqdm:
                # Load data and move to cuda
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                label_onehots = F.one_hot(labels, num_classes=10).float()
                # forward
                outputs = model(inputs)
                batch_loss = F.kl_div(F.log_softmax(outputs, dim = 1),
                                      label_onehots,
                                      reduction = 'sum')
                
                # store batch results
                epoch_loss += batch_loss * inputs.size(0)
                predictions = torch.max(outputs, dim = 1)[1]
                corrects += torch.sum(predictions == labels)
                testloader_tqdm.set_postfix({
                    'epoch_loss': epoch_loss.item(),
                    'corrects':   corrects.item()})
            
            if flag_baseline == True:
                return epoch_loss/test_loader.dataset.__len__(), corrects/test_loader.dataset.__len__()
            elif flag_baseline == False:
                teacher.metrics['test_loss'] = torch.cat([teacher.metrics['test_loss'], epoch_loss/test_loader.dataset.__len__()])
                teacher.metrics['test_accuracy'] = torch.cat([teacher.metrics['test_accuracy'], corrects/test_loader.dataset.__len__()])
                return teacher

    def training_loop_teacher(teacher, train_loader, test_loader):
        # Pre-allocate optimizer and scheduler
        teacher.optimizer = OPTIMIZER_FUNC(
            params = teacher.parameters(),
            lr = MAX_LEARNING_RATE,
            momentum = MOMENTUM,
            weight_decay = WEIGHT_DECAY)
        teacher.scheduler = SCHEDULER_FUNC(
            optimizer = teacher.optimizer,
            milestones = [100, 150],
            gamma = GAMMA)

        # Load state from checkpoint (best so far)
        try:
            teacher.metrics = torch.load('./results/CIFAR10/teacher_metrics.pth', map_location = DEVICE)
            state_dict = torch.load('./results/CIFAR10/teacher.pth')
            teacher.load_state_dict(state_dict["model_state_dict"])
            teacher.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            if teacher.scheduler is not None:
                teacher.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        except(FileNotFoundError):
            teacher.metrics = {
                "stopped": False,
                "stop_epoch": None,
                "cp_epoch": -1,
                'train_loss':     torch.tensor([], dtype = torch.float).to(DEVICE),
                'test_loss':      torch.tensor([], dtype = torch.float).to(DEVICE),
                'train_accuracy': torch.tensor([], dtype = torch.float).to(DEVICE),
                'test_accuracy':  torch.tensor([], dtype = torch.float).to(DEVICE)}
            print("Warning: File not found. Created new teacher and metrics.")
        finally:
            current_epoch = teacher.metrics["cp_epoch"] + 1
            if teacher.scheduler is not None:
                teacher.scheduler.last_epoch = current_epoch
        
        # Compute predicting baseline. Expected baseline accuracy = 10%
        # base_loss, base_accuracy = test(teacher, test_loader, flag_baseline = True)
        # print(f'Raw initialized teacher: Baseline loss = {base_loss.item():.0f}, accuracy = {base_accuracy.item():.4f}')

        # Begin training teacher (from cp_epoch)
        training_loop = tqdm(torch.arange(start = current_epoch, end = NUM_EPOCHS), desc = 'Training teacher', unit = 'epoch', leave = False)
        print(f"Continuing from epoch {current_epoch}.")
        for epoch in training_loop:
            if teacher.metrics["stopped"] == True:
                print(f'Stopped at epoch {epoch - 1}!')
                break

            train(teacher, train_loader)
            teacher.scheduler.step()
            test(teacher, test_loader)

            training_loop.set_postfix({
                'Train acc': teacher.metrics['train_accuracy'][epoch].item(),
                'Test acc': teacher.metrics['test_accuracy'][epoch].item(),
                'Train loss': teacher.metrics['train_loss'][epoch].item(),
                'Test loss': teacher.metrics['test_loss'][epoch].item()})

            # Early stopping: Non-improving test accuracy for NUM_EPOCHS_STOP epoch
            if epoch > NUM_EPOCHS_STOP:
                teacher.metrics["stopped"] = torch.prod(teacher.metrics['test_accuracy'][epoch - NUM_EPOCHS_STOP + 1:epoch + 1] < teacher.metrics['test_accuracy'][epoch - NUM_EPOCHS_STOP]).bool().item()

            # Save teacher when it improves
            if (teacher.metrics['test_accuracy'][epoch] == torch.max(teacher.metrics['test_accuracy'])):
                teacher.metrics["cp_epoch"] = epoch
                torch.save(obj = {'model_state_dict': teacher.state_dict(),
                                  'optimizer_state_dict': teacher.optimizer.state_dict(),
                                  'scheduler_state_dict': teacher.scheduler.state_dict() if teacher.scheduler else None,
                                  'best_test_accuracy': teacher.metrics['test_accuracy'][epoch]},
                           f = './results/CIFAR10/teacher.pth')
                print(f"Saved. Found best accuracy {teacher.metrics['test_accuracy'][epoch]} at epoch {epoch}.")
                torch.save(teacher.metrics, f = './results/CIFAR10/teacher_metrics.pth')
            # Save teacher when it stops
            if teacher.metrics["stopped"] == True:
                teacher.metrics["stop_epoch"] = epoch
                torch.save(teacher.metrics, f = './results/CIFAR10/teacher_metrics.pth')
                

        print(f"\nDone training teacher. Best accuracy: {torch.max(teacher.metrics['test_accuracy'], dim = 0)[0]} at epoch {torch.max(teacher.test_accuracy, dim = 0)[1]}")
        return teacher

    # Get teacher
    teacher = resnet20().to(DEVICE)
    teacher = training_loop_teacher(teacher, train_loader, test_loader)


# STAGE 2: KNOWLEDGE DISTILLATION
skip_distillation = False
if not skip_distillation:
    def train_students(students, teacher, train_loader):
        # Set mode for models, pre-allocate
        teacher.eval()
        for student in students:
            student.train()
            student.epoch_loss = torch.zeros([1]).to(DEVICE)
            student.corrects   = torch.zeros([1]).to(DEVICE)

        trainloader_tqdm = tqdm(train_loader, unit = 'batch', leave = False)
        trainloader_tqdm.set_description_str('Training students')
        for (inputs, labels) in trainloader_tqdm:
            # Load data and move to cuda
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            label_onehots = F.one_hot(labels, num_classes = 10).float()
            # Infer soft targets from teacher at TEMPERATURE as pseudo-labels, do not use ground truth
            with torch.no_grad():
                soft_targets = F.softmax(teacher(inputs)/TEMPERATURE, dim = 1)

            # forward
            for student in students:
                outputs = student(inputs)
                # knowledge distillation loss, at TEMPERATURE
                #   - F.kl_div() requires input as log-probabilities => feed through a log_softmax(), more
                #       optimized than log(softmax())
                #   - For probability distribution, KL Divergence loss gives similar effects to Cross-entropy
                #       (different by a constant: entropy of target)
                #   - Use F.kl_div(..., reduction= 'batchmean') to give mathematically correct result
                log_predictions_soft = F.log_softmax(outputs/TEMPERATURE, dim = 1)
                loss_distill = F.kl_div(log_predictions_soft, soft_targets, reduction = 'sum')
                # hard loss
                loss_hard = F.kl_div(F.log_softmax(outputs, dim = 1), label_onehots, reduction = 'sum')
                # overall loss = J1 + LAMBDA*J2
                # multiply J1 with TEMPERATURE^2 to rescale the gradients
                batch_loss = (1 - LAMBDA)*loss_distill + LAMBDA*loss_hard
                
                # backward
                student.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_value_(student.parameters(), GRAD_CLIP)
                student.optimizer.step()

                # store batch results
                student.epoch_loss += batch_loss * inputs.size(0)
                # predictions are based on student's ability only
                predictions = torch.max(outputs, dim = 1)[1]
                student.corrects += torch.sum(predictions == labels)

            trainloader_tqdm.set_postfix({
                'epoch_loss': torch.mean(torch.tensor([student.epoch_loss for student in students])).item(),
                'corrects': torch.mean(torch.tensor([student.corrects for student in students])).item()})

        for student in students:
            student.metrics['train_loss'] = torch.cat([student.metrics['train_loss'], student.epoch_loss/train_loader.dataset.__len__()])
            student.metrics['train_accuracy'] = torch.cat([student.metrics['train_accuracy'], student.corrects/train_loader.dataset.__len__()]) 

        return students

    def test_students(students, teacher, test_loader):
        # Set mode for models, pre-allocate
        teacher.eval()
        for student in students:
            student.eval()
            student.epoch_loss = torch.zeros([1]).to(DEVICE)
            student.corrects   = torch.zeros([1]).to(DEVICE)

        testloader_tqdm = tqdm(test_loader, unit = 'batch', leave = False)
        testloader_tqdm.set_description_str('Testing students')
        with torch.no_grad():
            for (inputs, labels) in testloader_tqdm:
                # Load data and move to cuda
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                label_onehots = F.one_hot(labels, num_classes = 10).float()
                # Infer soft targets from teacher at TEMPERATURE as pseudo-labels, do not use ground truth
                soft_targets = F.softmax(teacher(inputs)/TEMPERATURE, dim = 1)
                 
                for student in students:
                    # forward
                    outputs = student(inputs)

                    # knowledge distillation loss                
                    log_predictions_soft = F.log_softmax(outputs/TEMPERATURE, dim = 1)
                    loss_distill = F.kl_div(log_predictions_soft, soft_targets, reduction = 'sum')
                    # hard loss
                    loss_hard = F.kl_div(F.log_softmax(outputs, dim = 1), label_onehots, reduction = 'sum')     
                    # overall loss - same to train_students()
                    batch_loss = (1 - LAMBDA)*loss_distill + LAMBDA*loss_hard

                    # store batch results
                    student.epoch_loss += batch_loss * inputs.size(0)
                    # predictions are based on student's ability only
                    predictions = torch.max(outputs, dim = 1)[1]
                    student.corrects += torch.sum(predictions == labels)

                testloader_tqdm.set_postfix({
                    'epoch_loss': torch.mean(torch.tensor([student.epoch_loss for student in students])).item(),
                    'corrects': torch.mean(torch.tensor([student.corrects for student in students])).item()})

            for student in students:
                student.metrics['test_loss'] = torch.cat([student.metrics['test_loss'], student.epoch_loss/test_loader.dataset.__len__()])
                student.metrics['test_accuracy'] = torch.cat([student.metrics['test_accuracy'], student.corrects/test_loader.dataset.__len__()]) 

        return students

    def training_loop_distillation(students, teacher, train_loader, test_loader):
        print('Begin distillation.')
        # Pre-allocate students
        for student_id, student in enumerate(students):
            student.id = student_id
            student.optimizer = OPTIMIZER_FUNC(
                params = student.parameters(),
                amsgrad = True)
        try:
            for student in students:
                student.metrics = torch.load(f'./results/CIFAR10/student_{student.id}_metrics.pth', map_location = DEVICE)
                state_dict = torch.load(f'./results/CIFAR10/student_{student.id}_cp.pth', map_location = DEVICE)
                student.load_state_dict(state_dict["model_state_dict"])
                student.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                if student.scheduler is not None:
                    student.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        except (FileNotFoundError):
            for student in students:
                student.metrics = {
                    "stopped": False,
                    "stop_epoch": None,
                    "cp_epoch": -1,
                    "best_epoch": -1,
                    'train_loss':     torch.tensor([], dtype = torch.float).to(DEVICE),
                    'test_loss':      torch.tensor([], dtype = torch.float).to(DEVICE),
                    'train_accuracy': torch.tensor([], dtype = torch.float).to(DEVICE),
                    'test_accuracy':  torch.tensor([], dtype = torch.float).to(DEVICE)}
            print("Warning: File not found. Created new students and their metrics.")
        finally:
            attending_students = [student for student in students if (student.metrics["stopped"] == False)]
            assert len(attending_students) > 0, 'At least one student must attend.'
            
            cp_epochs = torch.tensor([student.metrics["cp_epoch"] for student in attending_students])
            assert torch.prod(((cp_epochs - cp_epochs[0]) == 0)).bool() == True, 'Checkpoint epochs should be equal.'
            
            current_epoch = attending_students[0].metrics["cp_epoch"] + 1
            for student in attending_students:
                if student.scheduler is not None:
                    student.scheduler.last_epoch = current_epoch

        training_loop = tqdm(torch.arange(start = current_epoch, end = NUM_EPOCHS), desc = 'Distillation', unit = 'epoch', leave = False)
        print(f"Continuing from epoch {current_epoch}.")
        for epoch in training_loop:
            attending_students = [student for student in students if (student.metrics["stopped"] == False)]

            # If all students has stopped => break
            if len(attending_students) == 0:
                print(f'All students stopped at epoch <= {epoch - 1}!')
                break

            train_students(attending_students, teacher, train_loader)
            for student in attending_students:
                if student.scheduler is not None:
                    student.scheduler.step()
            test_students(attending_students, teacher, test_loader)

            training_loop.set_postfix({
                'Train acc' : torch.mean(torch.tensor([student.metrics['train_accuracy'][epoch] for student in attending_students])).item(),
                'Test acc'  : torch.mean(torch.tensor([student.metrics['test_accuracy'][epoch] for student in attending_students])).item(),
                'Train loss': torch.mean(torch.tensor([student.metrics['train_loss'][epoch] for student in attending_students])).item(),
                'Test loss' : torch.mean(torch.tensor([student.metrics['test_loss'][epoch] for student in attending_students])).item()})
            
            flag_improve = False
            flag_early_stop = False
            for student in attending_students:
                # Student improves
                if student.metrics['test_accuracy'][epoch] == torch.max(student.metrics['test_accuracy']):
                    student.metrics["best_epoch"]  = epoch
                    print(f"Student {student.id} got new test accuracy {student.metrics['test_accuracy'][epoch].item()} at epoch {epoch}.")
                    torch.save(obj = {'model_state_dict': student.state_dict(),
                                      'best_test_accuracy': student.metrics['test_accuracy'][epoch]},
                               f = f'./results/CIFAR10/student_{student.id}.pth')
                    flag_improve = True
                # Student stops
                if epoch > NUM_EPOCHS_STOP:
                    student.metrics["stopped"] = torch.prod(student.metrics['test_accuracy'][epoch - NUM_EPOCHS_STOP + 1:epoch + 1] < student.metrics['test_accuracy'][epoch - NUM_EPOCHS_STOP]).bool().item()
                    if student.metrics["stopped"] == True:
                        student.metrics["stop_epoch"] = epoch
                        print(f'Student {student.id} stopped at epoch {epoch}.')
                        flag_early_stop = True

            # Checkpoint: Save all students when at least one improve or stops
            if flag_improve or flag_early_stop:
                for student in attending_students:
                    student.metrics["cp_epoch"] = epoch
                    torch.save(obj = {'model_state_dict': student.state_dict(),
                                      'optimizer_state_dict': student.optimizer.state_dict(),
                                      'scheduler_state_dict': student.scheduler.state_dict() if student.scheduler else None},
                               f = f'./results/CIFAR10/student_{student.id}_cp.pth')
                    torch.save(student.metrics, f'./results/CIFAR10/student_{student.id}_metrics.pth')
            
        print(f"Done distilling students. Their best accuracy: {[torch.max(student.metrics['test_accuracy']).item() for student in students]}")

    # Reload teacher (best checkpoint) from savefile
    if skip_training_teacher == True:
        teacher = resnet44().to(DEVICE)
        # Load pretrained teacher
        if FLAGS["USE_PRETRAINED"]:
            loaded = torch.load("./pretrained/resnet44-014dd654.th", map_location = DEVICE)
            
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in loaded['state_dict'].items():
                name = k[7:]                    # remove "module."
                new_state_dict[name] = v
            teacher.load_state_dict(new_state_dict)

            print(f"Loaded pretrained resnet44 teacher with {loaded['best_prec1']:.2f} accuracy.")
        # Load self-trained teacher
        else:
            loaded = torch.load('./results/CIFAR10/teacher.pth', map_location = DEVICE)
            teacher.load_state_dict(loaded['model_state_dict'])
            print(f"Loaded teacher, trained for {loaded['cp_epoch'] + 1} epochs, test accuracy {loaded['best_test_accuracy']}")
        
    # Init students
    students = [resnet20().to(DEVICE) for i in range(NUM_STUDENTS)]

    training_loop_distillation(students, teacher, train_loader, test_loader)


## STAGE 3: BASELINE FOR SELF-LEARNERS
skip_training_slearners = False
if not skip_training_slearners:
    def train_selflearners(slearners, train_loader):
        # Set mode for models, pre-allocate
        for slearner in slearners:
            slearner.train()
            slearner.epoch_loss = torch.zeros([1]).to(DEVICE)
            slearner.corrects   = torch.zeros([1]).to(DEVICE)

        trainloader_tqdm = tqdm(train_loader, unit = 'batch', leave = False)
        trainloader_tqdm.set_description_str('Training self-learners')
        for (inputs, labels) in trainloader_tqdm:
            # Load data and move to cuda
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            label_onehots = F.one_hot(labels, num_classes = 10).float()
            # forward
            for slearner in slearners:
                outputs = slearner(inputs)
                batch_loss = F.kl_div(F.log_softmax(outputs, dim = 1), label_onehots, reduction = 'sum')
                
                # backward
                slearner.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_value_(slearner.parameters(), GRAD_CLIP)
                slearner.optimizer.step()

                # store batch results
                slearner.epoch_loss += batch_loss * inputs.size(0)
                predictions = torch.max(outputs, dim = 1)[1]
                slearner.corrects += torch.sum(predictions == labels)

            trainloader_tqdm.set_postfix({
                'epoch_loss': torch.mean(torch.tensor([slearner.epoch_loss for slearner in slearners])).item(),
                'corrects': torch.mean(torch.tensor([slearner.corrects for slearner in slearners])).item()})

        for slearner in slearners:
            slearner.metrics['train_loss'] = torch.cat([slearner.metrics['train_loss'], slearner.epoch_loss/train_loader.dataset.__len__()])
            slearner.metrics['train_accuracy'] = torch.cat([slearner.metrics['train_accuracy'], slearner.corrects/train_loader.dataset.__len__()]) 

        return slearners

    def test_selflearners(slearners, test_loader):
        # Set mode for models, pre-allocate
        for slearner in slearners:
            slearner.eval()
            slearner.epoch_loss = torch.zeros([1]).to(DEVICE)
            slearner.corrects   = torch.zeros([1]).to(DEVICE)

        testloader_tqdm = tqdm(test_loader, unit = 'batch', leave = False)
        testloader_tqdm.set_description_str('Testing self-learners')
        with torch.no_grad():
            for (inputs, labels) in testloader_tqdm:
                # Load data and move to cuda
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                label_onehots = F.one_hot(labels, num_classes = 10).float()
                
                for slearner in slearners:
                    # forward
                    outputs = slearner(inputs)
                    batch_loss = F.kl_div(F.log_softmax(outputs, dim = 1), label_onehots, reduction = 'sum')     

                    # store batch results
                    slearner.epoch_loss += batch_loss * inputs.size(0)
                    predictions = torch.max(outputs, dim = 1)[1]
                    slearner.corrects += torch.sum(predictions == labels)

                testloader_tqdm.set_postfix({
                    'epoch_loss': torch.mean(torch.tensor([slearner.epoch_loss for slearner in slearners])).item(),
                    'corrects': torch.mean(torch.tensor([slearner.corrects for slearner in slearners])).item()})

            for slearner in slearners:
                slearner.metrics['test_loss'] = torch.cat([slearner.metrics['test_loss'], slearner.epoch_loss/test_loader.dataset.__len__()])
                slearner.metrics['test_accuracy'] = torch.cat([slearner.metrics['test_accuracy'], slearner.corrects/test_loader.dataset.__len__()]) 

        return slearners
    
    def training_loop_slearners(slearners, train_loader, test_loader):
        print('Begin training self-learners.')
        # Pre-allocate self-learners
        for slearner_id, slearner in enumerate(slearners):
            slearner.id = slearner_id
            slearner.optimizer = torch.optim.AdamW(
                params = slearner.parameters(),
                amsgrad = True)
            slearner.scheduler = None
            # slearner.scheduler = SCHEDULER_FUNC(
            #     optimizer = slearner.optimizer,
            #     milestones = [100, 150],
            #     gamma = GAMMA)
        try:
            for slearner in slearners:
                slearner.metrics = torch.load(f'./results/CIFAR10/slearner_{slearner.id}_metrics.pth', map_location = DEVICE)
                state_dict = torch.load(f'./results/CIFAR10/slearner_{slearner.id}_cp.pth', map_location = DEVICE)
                slearner.load_state_dict(state_dict["model_state_dict"])
                slearner.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                if slearner.scheduler is not None:
                    slearner.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        except (FileNotFoundError):
            for slearner in slearners:
                slearner.metrics = {
                    "stopped": False,
                    "stop_epoch": None,
                    "cp_epoch": -1,
                    "best_epoch": -1,
                    'train_loss':     torch.tensor([], dtype = torch.float).to(DEVICE),
                    'test_loss':      torch.tensor([], dtype = torch.float).to(DEVICE),
                    'train_accuracy': torch.tensor([], dtype = torch.float).to(DEVICE),
                    'test_accuracy':  torch.tensor([], dtype = torch.float).to(DEVICE)}
            print("Warning: File not found. Created new slearners and their metrics.")
        finally:
            attending_slearners = [slearner for slearner in slearners if (slearner.metrics["stopped"] == False)]
            assert len(attending_slearners) > 0, 'At least one slearner must attend.'
            
            cp_epochs = torch.tensor([slearner.metrics["cp_epoch"] for slearner in attending_slearners])
            assert torch.prod(((cp_epochs - cp_epochs[0]) == 0)).bool() == True, 'Checkpoint epochs should be equal.'
            
            current_epoch = attending_slearners[0].metrics["cp_epoch"] + 1
            for slearner in attending_slearners:
                if slearner.scheduler is not None:
                    slearner.scheduler.last_epoch = current_epoch

        training_loop = tqdm(torch.arange(start = current_epoch, end = NUM_EPOCHS), desc = 'Self-learning', unit = 'epoch', leave = False)
        print(f"Continuing from epoch {current_epoch}.")
        for epoch in training_loop:
            attending_slearners = [slearner for slearner in slearners if (slearner.metrics["stopped"] == False)]

            # If all self-learners has stopped => break
            if len(attending_slearners) == 0:
                print(f'All self-learners stopped at epoch <= {epoch - 1}!')
                break

            train_selflearners(attending_slearners, train_loader)
            for slearner in attending_slearners:
                if slearner.scheduler is not None:
                    slearner.scheduler.step()
            test_selflearners(attending_slearners, test_loader)

            training_loop.set_postfix({
                'Train acc' : torch.mean(torch.tensor([slearner.metrics['train_accuracy'][epoch] for slearner in attending_slearners])).item(),
                'Test acc'  : torch.mean(torch.tensor([slearner.metrics['test_accuracy'][epoch] for slearner in attending_slearners])).item(),
                'Train loss': torch.mean(torch.tensor([slearner.metrics['train_loss'][epoch] for slearner in attending_slearners])).item(),
                'Test loss' : torch.mean(torch.tensor([slearner.metrics['test_loss'][epoch] for slearner in attending_slearners])).item()})
            
            flag_improve = False
            flag_early_stop = False
            for slearner in attending_slearners:
                # Slearner improves
                if (slearner.metrics['test_accuracy'][epoch] == torch.max(slearner.metrics['test_accuracy'])):
                    slearner.metrics["best_epoch"]  = epoch
                    print(f"slearner {slearner.id} got new test accuracy {slearner.metrics['test_accuracy'][epoch].item()} at epoch {epoch}.")
                    torch.save(obj = {'model_state_dict': slearner.state_dict(),
                                      'best_test_accuracy': slearner.metrics['test_accuracy'][epoch]},
                               f = f'./results/CIFAR10/slearner_{slearner.id}.pth')
                    flag_improve = True
                # Slearner stops
                if epoch > NUM_EPOCHS_STOP:
                    slearner.metrics["stopped"] = torch.prod(slearner.metrics['test_accuracy'][epoch - NUM_EPOCHS_STOP + 1:epoch + 1] < slearner.metrics['test_accuracy'][epoch - NUM_EPOCHS_STOP]).bool().item()
                    if slearner.metrics["stopped"] == True:
                        slearner.metrics["stop_epoch"] = epoch
                        print(f'slearner {slearner.id} stopped at epoch {epoch}.')
                        flag_early_stop = True
                

            # Checkpoint: Save all students when at least one improve or stops
            if flag_improve or flag_early_stop or epoch == NUM_EPOCHS - 1:
                for slearner in attending_slearners:
                    slearner.metrics["cp_epoch"] = epoch
                    torch.save(obj = {'model_state_dict': slearner.state_dict(),
                                      'optimizer_state_dict': slearner.optimizer.state_dict(),
                                      'scheduler_state_dict': slearner.scheduler.state_dict() if slearner.scheduler else None},
                               f = f'./results/CIFAR10/slearner_{slearner.id}_cp.pth')
                    torch.save(slearner.metrics, f'./results/CIFAR10/slearner_{slearner.id}_metrics.pth')
            
        print(f"Done training self-learners. Their best accuracy: {[torch.max(slearner.metrics['test_accuracy']).item() for slearner in slearners]}")

    # Init self-learners
    slearners = [resnet20().to(DEVICE) for i in range(NUM_STUDENTS)]
    training_loop_slearners(slearners, train_loader, test_loader)