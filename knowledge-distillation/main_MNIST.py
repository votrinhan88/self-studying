import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import Teacher, Student
from dataset import Dataset

LEARNING_RATE = 0.005
MOMENTUM = 0.7
NUM_STUDENTS = 5
NUM_EPOCHS = 20
NUM_EPOCHS_STOP = 2
TEMPERATURE = 4
LAMBDA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # cpu on my laptop

# Get dataloaders
dataset = Dataset()
train_loader, test_loader = dataset.get_MNIST()

# STAGE 1: TRAIN TEACHER
# Init teacher
skip_training_teacher = False
if not skip_training_teacher:
    def train(model, train_loader):
        model.train()

        epoch_loss = torch.tensor([0], dtype = torch.float)
        corrects = torch.tensor([0])

        trainloader_tqdm = tqdm(train_loader, unit = 'batch', leave = False)
        trainloader_tqdm.set_description_str('Training teacher')
        for (inputs, labels) in trainloader_tqdm:
            # Load data and move to cuda
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # forward
            outputs = model(inputs)
            batch_loss = F.cross_entropy(outputs, labels)
            predictions = torch.max(outputs, dim = 1)[1]
            # backward
            teacher.optimizer.zero_grad()
            batch_loss.backward()
            teacher.optimizer.step()

            # store batch results
            epoch_loss += batch_loss * inputs.size(0)
            corrects += torch.sum(predictions == labels)
            trainloader_tqdm.set_postfix({'epoch_loss': epoch_loss.item(),
                                        'corrects': corrects.item()})

        teacher.train_loss = torch.cat([teacher.train_loss, epoch_loss])
        teacher.train_accuracy = torch.cat([teacher.train_accuracy, corrects/train_loader.dataset.__len__()])

    def test(model, test_loader):
        model.eval()
        epoch_loss = torch.tensor([0], dtype = torch.float)
        corrects = torch.tensor([0])

        testloader_tqdm = tqdm(test_loader, unit = 'batch', leave = False)
        testloader_tqdm.set_description_str('Testing teacher')
        with torch.no_grad():
            for (inputs, labels) in testloader_tqdm:
                # Load data and move to cuda
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                # forward
                outputs = model(inputs)
                softmaxs = F.softmax(outputs, dim = 1)
                batch_loss = F.cross_entropy(softmaxs, labels)
                
                # store batch results
                epoch_loss += batch_loss * inputs.size(0)
                predictions = torch.max(outputs, dim = 1)[1]
                corrects += torch.sum(predictions == labels)
                testloader_tqdm.set_postfix({'epoch_loss': epoch_loss.item(),
                                            'corrects': corrects.item()})

            teacher.test_loss = torch.cat([teacher.test_loss, epoch_loss])
            teacher.test_accuracy = torch.cat([teacher.test_accuracy, corrects/test_loader.dataset.__len__()])

    def training_loop_teacher(teacher, train_loader, test_loader):
        teacher.optimizer = torch.optim.SGD(teacher.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

        # Compute predicting baseline. Expected baseline accuracy = 10%
        teacher.test_loss = torch.tensor([], dtype = torch.float)
        teacher.test_accuracy = torch.tensor([], dtype = torch.float)
        test(teacher, test_loader)
        print(f'Raw initialized teacher: Baseline accuracy = {teacher.test_accuracy.item():.4f}')

        # Pre-allocate
        teacher.train_loss = torch.tensor([], dtype = torch.float)
        teacher.test_loss = torch.tensor([], dtype = torch.float)
        teacher.train_accuracy = torch.tensor([], dtype = torch.float)
        teacher.test_accuracy = torch.tensor([], dtype = torch.float)
        teacher.stopped = torch.tensor([False])

        # Begin training teacher
        training_loop = tqdm(torch.arange(NUM_EPOCHS), desc = 'Training loop', unit = 'epoch', leave = False)
        for epoch in training_loop:
            if teacher.stopped == True:
                print(f'Stopped at epoch {epoch - 1}!')
                break

            train(teacher, train_loader)
            test(teacher, test_loader)

            training_loop.set_postfix({
                'Train acc': teacher.train_accuracy[epoch].item(),
                'Test acc': teacher.test_accuracy[epoch].item(),
                'Train loss': teacher.train_loss[epoch].item(),
                'Test loss': teacher.test_loss[epoch].item()})

            # Early stopping: Non-improving test accuracy for NUM_EPOCHS_STOP epoch
            if epoch > NUM_EPOCHS_STOP:
                teacher.stopped = torch.prod(teacher.test_accuracy[epoch - NUM_EPOCHS_STOP + 1:epoch + 1] < teacher.test_accuracy[epoch - NUM_EPOCHS_STOP]).bool().item()

            # Save teacher when it improves
            if (teacher.test_accuracy[epoch] == torch.max(teacher.test_accuracy)):
                torch.save(obj = {'model_state_dict': teacher.state_dict(),
                                  'optimizer_state_dict': teacher.optimizer.state_dict(),
                                  'best_epoch': epoch,
                                  'best_test_accuracy': teacher.test_accuracy[epoch]},
                           f = './results/MNIST/teacher_extra.pth')
                # print(f'Saved. Found best accuracy {test_accuracy[epoch]} at epoch {epoch}.')

            if teacher.stopped == True:
                torch.save(obj = {'stop_epoch': epoch,
                                  'train_accuracy': teacher.train_accuracy,
                                  'train_loss': teacher.train_loss,
                                  'test_accuracy': teacher.test_accuracy,
                                  'test_loss': teacher.test_loss},
                           f = './results/MNIST/teacher_extra_metrics.pth')

        print(f'\nDone training teacher. Best accuracy: {torch.max(teacher.test_accuracy, dim = 0)[0]} at epoch {torch.max(teacher.test_accuracy, dim = 0)[1]}')
        return teacher

    teacher = Teacher()
    training_loop_teacher(teacher, train_loader, test_loader)
    

# STAGE 2: KNOWLEDGE DISTILLATION
skip_teaching_students = False
if not skip_teaching_students:
    def train_students(students, teacher, train_loader):
        # Set mode for models, pre-allocate
        teacher.eval()
        for student in students:
            student.train()
            student.epoch_loss = torch.zeros(1)
            student.corrects   = torch.zeros(1)

        trainloader_tqdm = tqdm(train_loader, unit = 'batch', leave = False)
        trainloader_tqdm.set_description_str('Training students')
        for (inputs, labels) in trainloader_tqdm:
            # Load data and move to cuda
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
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
                loss_distill = F.kl_div(log_predictions_soft, soft_targets, reduction = 'batchmean')
                # hard loss
                loss_hard = F.cross_entropy(outputs, labels)            
                # overall loss = J1 + LAMBDA*J2
                # multiply J1 with TEMPERATURE^2 to rescale the gradients
                batch_loss = loss_distill*(TEMPERATURE**2) + LAMBDA*loss_hard
                
                # backward
                student.optimizer.zero_grad()
                batch_loss.backward()
                student.optimizer.step()

                # predictions are based on student's ability only
                predictions = torch.max(outputs, dim = 1)[1]
                # store batch results
                student.epoch_loss += batch_loss * inputs.size(0)
                student.corrects += torch.sum(predictions == labels)

            trainloader_tqdm.set_postfix({
                'epoch_loss': torch.mean(torch.tensor([student.epoch_loss for student in students])).item(),
                'corrects': torch.mean(torch.tensor([student.corrects for student in students])).item()})

        for student in students:
            student.accuracy = student.corrects/train_loader.dataset.__len__()

        return students

    def test_students(students, teacher, test_loader):
        # Set mode for models, pre-allocate
        teacher.eval()
        for student in students:
            student.eval()
            student.epoch_loss = torch.zeros(1)
            student.corrects   = torch.zeros(1)

        testloader_tqdm = tqdm(test_loader, unit = 'batch', leave = False)
        testloader_tqdm.set_description_str('Testing students')
        with torch.no_grad():
            for (inputs, labels) in testloader_tqdm:
                # Load data and move to cuda
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                # Infer soft targets from teacher at TEMPERATURE as pseudo-labels, do not use ground truth
                soft_targets = F.softmax(teacher(inputs)/TEMPERATURE, dim = 1)
                
                # forward
                for student_id, student in enumerate(students):
                    outputs = student(inputs)

                    # knowledge distillation loss                
                    log_predictions_soft = F.log_softmax(outputs/TEMPERATURE, dim = 1)
                    loss_distill = F.kl_div(log_predictions_soft, soft_targets, reduction = 'batchmean')
                    # hard loss
                    loss_hard = F.cross_entropy(outputs, labels)     
                    # overall loss - same to train_students()
                    batch_loss = loss_distill*(TEMPERATURE**2) + LAMBDA*loss_hard

                    # predictions are based on student's ability only
                    predictions = torch.max(outputs, dim = 1)[1]
                    # store batch results
                    student.epoch_loss += batch_loss * inputs.size(0)
                    student.corrects += torch.sum(predictions == labels)

                testloader_tqdm.set_postfix({
                    'epoch_loss': torch.mean(torch.tensor([student.epoch_loss for student in students])).item(),
                    'corrects': torch.mean(torch.tensor([student.corrects for student in students])).item()})

            for student in students:
                student.accuracy = student.corrects/test_loader.dataset.__len__()

        return students

    # Reload teacher (best checkpoint) from savefile
    if skip_training_teacher == True:
        teacher = Teacher()
        loaded = torch.load('./results/MNIST/teacher.pth')
        teacher.load_state_dict(loaded['model_state_dict'])
        print(f"Loaded teacher, trained for {loaded['best_epoch'] + 1} epochs, test accuracy {loaded['best_test_accuracy']}")

    print('Begin teaching students.')

    # Init students & pre-allocate
    students = [Student() for i in range(NUM_STUDENTS)]
    for student_id, student in enumerate(students):
        student.id = student_id
        student.optimizer = torch.optim.SGD(student.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
        student.train_loss     = torch.zeros([NUM_EPOCHS])
        student.test_loss      = torch.zeros([NUM_EPOCHS])
        student.train_accuracy = torch.zeros([NUM_EPOCHS])
        student.test_accuracy  = torch.zeros([NUM_EPOCHS])
        student.stopped = False

    training_loop = tqdm(torch.arange(NUM_EPOCHS), desc = 'Training loop', unit = 'epoch', leave = False)
    # training_loop = torch.arange(NUM_EPOCHS)
    for epoch in training_loop:
        attending_students = [student for student in students if (student.stopped == False)]

        # If all students has stopped => break
        if len(attending_students) == 0:
            print(f'All students stopped at epoch <= {epoch - 1}!')
            break

        train_students(attending_students, teacher, train_loader)
        for student in attending_students:
            student.train_accuracy[epoch] = student.accuracy
            student.train_loss[epoch]     = student.epoch_loss
        test_students(attending_students, teacher, test_loader)
        for student in attending_students:
            student.test_accuracy[epoch] = student.accuracy
            student.test_loss[epoch]     = student.epoch_loss

        training_loop.set_postfix({
            'Train acc' : torch.mean(torch.tensor([student.train_accuracy[epoch] for student in attending_students])).item(),
            'Test acc'  : torch.mean(torch.tensor([student.test_accuracy[epoch] for student in attending_students])).item(),
            'Train loss': torch.mean(torch.tensor([student.train_loss[epoch] for student in attending_students])).item(),
            'Test loss' : torch.mean(torch.tensor([student.test_loss[epoch] for student in attending_students])).item()})
        
        for student in attending_students:
            # Early stopping
            if epoch > NUM_EPOCHS_STOP:
                student.stopped = torch.prod(student.test_accuracy[epoch - NUM_EPOCHS_STOP + 1:epoch + 1] < student.test_accuracy[epoch - NUM_EPOCHS_STOP]).bool().item()
                if student.stopped:
                    print(f'Student {student.id} stopped at epoch {epoch}.')
            # Save students when they improve
            if (student.test_accuracy[epoch] == torch.max(student.test_accuracy)):
                torch.save(obj = {'model_state_dict': student.state_dict(),
                                'optimizer_state_dict': student.optimizer.state_dict(),
                                'best_epoch': epoch,
                                'best_test_accuracy': student.test_accuracy[epoch]},
                        f = f'./results/MNIST/student-{student.id}.pth')
                print(f'Saved. Student {student.id} got new test accuracy {student.test_accuracy[epoch].item()} at epoch {epoch}.')

            if student.stopped == True:
                torch.save(obj = {'stop_epoch': epoch,
                                'train_accuracy': student.train_accuracy,
                                'train_loss': student.train_loss,
                                'test_accuracy': student.test_accuracy,
                                'test_loss': student.test_loss},
                        f = f'./results/MNIST/student-{student.id}_metrics.pth')
        
    print(f'Done distilling students. Their best accuracy: {[torch.max(student.test_accuracy).item() for student in students]}')


## STAGE 3: BASELINE FOR SELF-LEARNERS
def train_selflearners(students, train_loader):
    # Set mode for models, pre-allocate
    for student in students:
        student.train()
        student.epoch_loss = torch.zeros(1)
        student.corrects   = torch.zeros(1)

    trainloader_tqdm = tqdm(train_loader, unit = 'batch', leave = False)
    trainloader_tqdm.set_description_str('Training self-learners')
    for (inputs, labels) in trainloader_tqdm:
        # Load data and move to cuda
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward
        for student in students:
            outputs = student(inputs)
            batch_loss = F.cross_entropy(outputs, labels)
            
            # backward
            student.optimizer.zero_grad()
            batch_loss.backward()
            student.optimizer.step()

            predictions = torch.max(outputs, dim = 1)[1]
            # store batch results
            student.epoch_loss += batch_loss * inputs.size(0)
            student.corrects += torch.sum(predictions == labels)

        trainloader_tqdm.set_postfix({
            'epoch_loss': torch.mean(torch.tensor([student.epoch_loss for student in students])).item(),
            'corrects': torch.mean(torch.tensor([student.corrects for student in students])).item()})

    for student in students:
        student.accuracy = student.corrects/train_loader.dataset.__len__()

    return students

def test_selflearners(students, test_loader):
    # Set mode for models, pre-allocate
    for student in students:
        student.eval()
        student.epoch_loss = torch.zeros(1)
        student.corrects   = torch.zeros(1)

    testloader_tqdm = tqdm(test_loader, unit = 'batch', leave = False)
    testloader_tqdm.set_description_str('Testing students')
    with torch.no_grad():
        for (inputs, labels) in testloader_tqdm:
            # Load data and move to cuda
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # forward
            for student in students:
                outputs = student(inputs)
                batch_loss = F.cross_entropy(outputs, labels)     

                predictions = torch.max(outputs, dim = 1)[1]
                # store batch results
                student.epoch_loss += batch_loss * inputs.size(0)
                student.corrects += torch.sum(predictions == labels)

            testloader_tqdm.set_postfix({
                'epoch_loss': torch.mean(torch.tensor([student.epoch_loss for student in students])).item(),
                'corrects': torch.mean(torch.tensor([student.corrects for student in students])).item()})

        for student in students:
            student.accuracy = student.corrects/test_loader.dataset.__len__()

    return students

# Reload teacher (best checkpoint) from savefile
print('Begin training self-learners.')

# Init students & pre-allocate
selflearners = [Student() for i in range(NUM_STUDENTS)]
for student_id, student in enumerate(selflearners):
    student.id = student_id
    student.optimizer = torch.optim.SGD(student.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    student.train_loss     = torch.zeros([NUM_EPOCHS])
    student.test_loss      = torch.zeros([NUM_EPOCHS])
    student.train_accuracy = torch.zeros([NUM_EPOCHS])
    student.test_accuracy  = torch.zeros([NUM_EPOCHS])
    student.stopped = False

training_loop = tqdm(torch.arange(NUM_EPOCHS), desc = 'Training loop', unit = 'epoch', leave = False)
for epoch in training_loop:
    attending_students = [student for student in selflearners if (student.stopped == False)]

    # If all students has stopped => break
    if len(attending_students) == 0:
        print(f'All students stopped at epoch <= {epoch - 1}!')
        break

    train_selflearners(attending_students, train_loader)
    for student in attending_students:
        student.train_accuracy[epoch] = student.accuracy
        student.train_loss[epoch]     = student.epoch_loss
    test_selflearners(attending_students, test_loader)
    for student in attending_students:
        student.test_accuracy[epoch] = student.accuracy
        student.test_loss[epoch]     = student.epoch_loss

    training_loop.set_postfix({
        'Train acc' : torch.mean(torch.tensor([student.train_accuracy[epoch] for student in attending_students])).item(),
        'Test acc'  : torch.mean(torch.tensor([student.test_accuracy[epoch] for student in attending_students])).item(),
        'Train loss': torch.mean(torch.tensor([student.train_loss[epoch] for student in attending_students])).item(),
        'Test loss' : torch.mean(torch.tensor([student.test_loss[epoch] for student in attending_students])).item()})
    
    for student in attending_students:
        # Early stopping
        if epoch > NUM_EPOCHS_STOP:
            student.stopped = torch.prod(student.test_accuracy[epoch - NUM_EPOCHS_STOP + 1:epoch + 1] < student.test_accuracy[epoch - NUM_EPOCHS_STOP]).bool().item()
            if student.stopped:
                print(f'Student {student.id} stopped at epoch {epoch}.')
        # Save students when they improve
        if (student.test_accuracy[epoch] == torch.max(student.test_accuracy)):
            torch.save(obj = {'model_state_dict': student.state_dict(),
                            'optimizer_state_dict': student.optimizer.state_dict(),
                            'best_epoch': epoch,
                            'best_test_accuracy': student.test_accuracy[epoch]},
                    f = f'./results/MNIST/selflearner-{student.id}.pth')
            print(f'Saved. Student {student.id} got new test accuracy {student.test_accuracy[epoch].item()} at epoch {epoch}.')

        if student.stopped == True:
            torch.save(obj = {'stop_epoch': epoch,
                            'train_accuracy': student.train_accuracy,
                            'train_loss': student.train_loss,
                            'test_accuracy': student.test_accuracy,
                            'test_loss': student.test_loss},
                    f = f'./results/MNIST/selflearner-{student.id}_metrics.pth')
    
print(f'Done training self-learners. Their best accuracy: {[torch.max(student.test_accuracy).item() for student in selflearners]}')