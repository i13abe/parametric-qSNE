import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, RandomSampler

import numpy as np

import os
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces

#The dataset for tiny imagenet 
class TinyImagenet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        
        root = os.path.expanduser(root)
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
            
        self.target_transform = target_transform
            
        if train:
            self.data = np.load(root + '/train/img_train.npy')
            self.label = np.load(root + '/train/label_train.npy').astype(np.int64)
            self.label = torch.from_numpy(self.label)
            self.num = len(self.data)
        else:
            self.data = np.load(root + '/test/img_test.npy')
            self.label = np.load(root + '/test/label_test.npy').astype(np.int64)
            self.label = torch.from_numpy(self.label)
            self.num = len(self.data)
        
        self.data = np.uint8(self.data)
        self.classes = ['goldfish, Carassius auratus', 'European fire salamander, Salamandra salamandra',
                        'bullfrog, Rana catesbeiana', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
                        'American alligator, Alligator mississipiensis', 'boa constrictor, Constrictor constrictor',
                        'trilobite', 'scorpion', 'black widow, Latrodectus mactans', 'tarantula', 'centipede',
                        'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'jellyfish',
                        'brain coral', 'snail', 'slug', 'sea slug, nudibranch', 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
                        'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'black stork, Ciconia nigra',
                        'king penguin, Aptenodytes patagonica', 'albatross, mollymawk', 'dugong, Dugong dugon',
                        'Chihuahua', 'Yorkshire terrier', 'golden retriever', 'Labrador retriever',
                        'German shepherd, German shepherd dog, German police dog, alsatian', 'standard poodle',
                        'tabby, tabby cat', 'Persian cat', 'Egyptian cat', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
                        'lion, king of beasts, Panthera leo', 'brown bear, bruin, Ursus arctos',
                        'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'fly', 'bee',
                        'grasshopper, hopper', 'walking stick, walkingstick, stick insect', 'cockroach, roach',
                        'mantis, mantid', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
                        'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'sulphur butterfly, sulfur butterfly',
                        'sea cucumber, holothurian', 'guinea pig, Cavia cobaya', 'hog, pig, grunter, squealer, Sus scrofa',
                        'ox', 'bison', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
                        'gazelle', 'Arabian camel, dromedary, Camelus dromedarius', 'orangutan, orang, orangutang, Pongo pygmaeus',
                        'chimpanzee, chimp, Pan troglodytes', 'baboon', 'African elephant, Loxodonta africana',
                        'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'abacus',
                        "academic gown, academic robe, judge's robe", 'altar', 'apron',
                        'backpack, back pack, knapsack, packsack, rucksack, haversack', 'bannister, banister, balustrade, balusters, handrail',
                        'barbershop', 'barn', 'barrel, cask', 'basketball', 'bathtub, bathing tub, bath, tub',
                        'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
                        'beacon, lighthouse, beacon light, pharos', 'beaker', 'beer bottle', 'bikini, two-piece',
                        'binoculars, field glasses, opera glasses', 'birdhouse', 'bow tie, bow-tie, bowtie',
                        'brass, memorial tablet, plaque', 'broom', 'bucket, pail', 'bullet train, bullet',
                        'butcher shop, meat market', 'candle, taper, wax light', 'cannon', 'cardigan',
                        'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
                        'CD player', 'chain', 'chest', 'Christmas stocking', 'cliff dwelling', 'computer keyboard, keypad',
                        'confectionery, confectionary, candy store', 'convertible', 'crane', 'dam, dike, dyke',
                        'desk', 'dining table, board', 'drumstick', 'dumbbell', 'flagpole, flagstaff', 'fountain',
                        'freight car', 'frying pan, frypan, skillet', 'fur coat', 'gasmask, respirator, gas helmet',
                        'go-kart', 'gondola', 'hourglass', 'iPod', 'jinrikisha, ricksha, rickshaw', 'kimono',
                        'lampshade, lamp shade', 'lawn mower, mower', 'lifeboat', 'limousine, limo', 'magnetic compass',
                        'maypole', 'military uniform', 'miniskirt, mini', 'moving van', 'nail', 'neck brace',
                        'obelisk', 'oboe, hautboy, hautbois', 'organ, pipe organ', 'parking meter', 'pay-phone, pay-station',
                        'picket fence, paling', 'pill bottle', "plunger, plumber's helper", 'pole',
                        'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho',
                        'pop bottle, soda bottle', "potter's wheel", 'projectile, missile', 'punching bag, punch bag, punching ball, punchball',
                        'reel', 'refrigerator, icebox', 'remote control, remote', 'rocking chair, rocker',
                        'rugby ball', 'sandal', 'school bus', 'scoreboard', 'sewing machine', 'snorkel', 'sock',
                        'sombrero', 'space heater', "spider web, spider's web", 'sports car, sport car',
                        'steel arch bridge', 'stopwatch, stop watch', 'sunglasses, dark glasses, shades',
                        'suspension bridge', 'swimming trunks, bathing trunks', 'syringe', 'teapot', 'teddy, teddy bear',
                        'thatch, thatched roof', 'torch', 'tractor', 'triumphal arch', 'trolleybus, trolley coach, trackless trolley',
                        'turnstile', 'umbrella', 'vestment', 'viaduct', 'volleyball', 'water jug', 'water tower',
                        'wok', 'wooden spoon', 'comic book', 'plate', 'guacamole', 'ice cream, icecream',
                        'ice lolly, lolly, lollipop, popsicle', 'pretzel', 'mashed potato', 'cauliflower',
                        'bell pepper', 'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', 'meat loaf, meatloaf',
                        'pizza, pizza pie', 'potpie', 'espresso', 'alp', 'cliff, drop, drop-off', 'coral reef',
                        'lakeside, lakeshore', 'seashore, coast, seacoast, sea-coast', 'acorn']
        
        
    def __len__(self):
        return self.num
        
        
    def __getitem__(self, idx):
        trans = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])
        out_data = trans(self.data[idx])
        out_label = self.label[idx]

        out_data = self.transform(out_data)
        
        if self.target_transform is not None:
            out_label = self.target_transform(out_label)

        return out_data, out_label

    
#The dataset for olivetti faces
class OlivettiFaces(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
            
        self.data = fetch_olivetti_faces(data_home = root)
        self.num = len(self.data.data)
        
    def __len__(self):
        return self.num
        
        
    def __getitem__(self, idx):
        out_data = self.transform(self.data.images[idx])
        out_label = self.data.target[idx]

        return out_data, out_label

class Glove(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        
        root = os.path.expanduser(root)
        
        if transform is None:
            self.transform = torch.tensor
        else:
            self.transform = transform
            
        self.data = np.load(root + '/data.npy')[:10000]
        f = open(root + '/label.txt')
        labels = f.readline()
        labels = labels.rstrip('\n').split(' ')
        self.labels = labels[:10000]
        self.num = len(self.data)
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        out_data = self.transform(self.data[idx])
        out_label = self.labels[idx]
        
        return out_data, out_label
    
    
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = dataset.classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return idx, self.dataset[idx][0], self.dataset[idx][1]
        
    
    
class Datasets(object):
    def __init__(self,
                 dataset_name,
                 batch_size=100,
                 num_workers=2,
                 transform=None,
                 test_transform="same",
                 shuffle = True):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle
        if test_transform is "same":
            self.test_transform = transform
        else:
            self.test_transform = test_transform
            
        if self.transform is None:
                self.transform = transforms.Compose([transforms.ToTensor()])
        if self.test_transform is None:
                self.test_transform = transforms.Compose([transforms.ToTensor()])
        
        
    def create(self):
        print("Dataset :",self.dataset_name)
        
        traindata, classes, base_labels, input_channels = self.set_dataset(train=True)
        testdata, _, _, _ = self.set_dataset(train=False)
            
        
        trainloader = self.set_dataloader(traindata)
        
        if testdata is not None:
            testloader = self.set_dataloader(testdata,
                                             shuffle=False)
        else:
            testloader = None
            
        return [trainloader, testloader, classes, base_labels, input_channels, traindata, testdata]
    
    
    def set_dataloader(self, dataset, num_workers=None, batch_size=None, shuffle=None, batch_sampler=None):
        if num_workers is None:
            num_workers = self.num_workers
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle
        
        if batch_sampler is not None:
            dataloader  =torch.utils.data.DataLoader(dataset,
                                                    batch_sampler = batch_sampler,
                                                    num_workers=num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                         num_workers=num_workers)
        return dataloader
    
    def set_dataset(self, train=True, trans=None):
        if train:
            print("set train data")
            transform = self.transform
        else:
            print("set test data")
            transform = self.test_transform
        
        if trans is not None:
            transform = trans
        
        if self.dataset_name == "MNIST":
            path = '~/work/MNISTDataset/data'
            dataset = torchvision.datasets.MNIST(root=path,
                                                 train=train,
                                                 download=True,
                                                 transform=transform)
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 1
            
        elif self.dataset_name == "FashionMNIST":
            path = '~/work/FashionMNISTDataset/data'
            dataset = torchvision.datasets.FashionMNIST(root=path,
                                                        train=train,
                                                        download=True,
                                                        transform=transform)
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 1
            
        elif self.dataset_name == "CIFAR10":
            path = '~/work/CIFAR10Dataset/data'
            dataset = torchvision.datasets.CIFAR10(root=path,
                                                   train=train,
                                                   download=True,
                                                   transform=transform)
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 3
            
        elif self.dataset_name == "CIFAR100":
            path = '~/work/CIFAR100Dataset/data'
            dataset = torchvision.datasets.CIFAR100(root=path,
                                                    train=train,
                                                    download=True,
                                                    transform=transform)
            classes = list(range(100))
            base_labels = dataset.classes
            input_channels = 3
            
        elif self.dataset_name == "STL10":
            path = '~/work/STL10/data'
            if train:
                split = "train"
            else:
                split = "test"
            dataset = torchvision.datasets.STL10(root=path,
                                                 split=split,
                                                 download=True,
                                                 transform=transform)
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 3
        
        elif self.dataset_name == "TinyImagenet":
            path = '~/work/TinyImagenet'
            dataset = TinyImagenet(root=path,
                                   train=train,
                                   transform=transform)
            classes = list(range(200))
            base_labels = dataset.classes
            input_channels = 3
            
        elif self.dataset_name == "OlivettiFaces":
            path = '~/work/Olivettifaces/data'
            dataset = OlivettiFaces(root=path,
                                        transform=transform)
            classes = list(range(40))
            base_labels = []
            input_channels = 1
            if not train:
                dataset = None
            
        elif self.dataset_name == "COIL-20":
            path = '~/work/COIL-20/data'
            transform = transforms.Compose([transforms.Grayscale(),
                                            transforms.ToTensor()])
            dataset = torchvision.datasets.ImageFolder(root=path,
                                                           transform=transform)
            classes = list(range(20))
            base_labels = dataset.classes
            input_channels = 1
            if not train:
                dataset = None
        
        elif self.dataset_name == "Glove":
            path = '~/work/Glove/data'
            dataset = Glove(root=path,
                                transform=None)
            classes = list(range(len(dataset)))
            base_labels = dataset.labels
            input_channels = 1
            if not train:
                dataset = None
            
        elif self.dataset_name == "VGGface":
            if train:
                path = '~/work/VGGface2/train'
            else:
                path = '~/work/VGGface2/test'
            dataset = torchvision.datasets.ImageFolder(root=path,
                                                       transform=transform)
            classes = None
            base_labels = None
            input_channels = 3
        
        elif self.dataset_name == "FractalDB-60":
            path = '~/work/FractalDB-60/data'
            dataset = torchvision.datasets.ImageFolder(root=path,
                                                           transform=transform)
            classes = list(range(60))
            base_labels = dataset.classes
            input_channels = 1
            if not train:
                dataset = None
        
        elif self.dataset_name == "FractalDB-1k":
            path = '~/work/FractalDB-1k/data'
            dataset = torchvision.datasets.ImageFolder(root=path,
                                                           transform=transform)
            classes = list(range(1000))
            base_labels = dataset.classes
            input_channels = 1
            if not train:
                dataset = None
            
        else:
            raise KeyError("Unknown dataset: {}".format(self.dataset_name))
            
        return dataset, classes, base_labels, input_channels
        
    
    def worker_init_fn(self, worker_id):                                                          
        np.random.seed(worker_id)