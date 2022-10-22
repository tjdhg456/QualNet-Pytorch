from module.backbone.resnet import load_resnet
from module.backbone.irevnet import iRevNet
from module.backbone.wrapper import Wrapper

def load_teacher_model(args):
    # Backbone
    if args.backbone == 'resnet18':
        model = load_resnet(depth=18, num_classes=args.num_classes, attention_type=args.mode, imagenet=args.data_type == 'imagenet')
    elif args.backbone == 'resnet50':
        model = load_resnet(depth=50, num_classes=args.num_classes, attention_type=args.mode, imagenet=args.data_type == 'imagenet')
    else:
        raise('Select Proper Backbone')
    
    
    # Decoder
    decoder = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                  nChannels=[24, 96, 384, 1536], nClasses=1000, init_ds=2,
                  dropout_rate=0., affineBN=True, in_shape=[3, 256, 256], mult=4)
    
    model_wrapper = Wrapper(model=model, decoder=decoder)
    return model_wrapper
        

def load_student_model(args):
    # Load Student
    if args.backbone == 'resnet18':
        model = load_resnet(depth=18, num_classes=args.num_classes, attention_type=args.mode, imagenet=args.data_type == 'imagenet')
    elif args.backbone == 'resnet50':
        model = load_resnet(depth=50, num_classes=args.num_classes, attention_type=args.mode, imagenet=args.data_type == 'imagenet')
    elif args.backbone == 'resnet101':
        model = load_resnet(depth=101, num_classes=args.num_classes, attention_type=args.mode, imagenet=args.data_type == 'imagenet')
    else:
        raise('Select Proper Backbone')
    
    
    # Load Pre-trained Teacher    
    if args.data_type == 'imagenet':
        teacher_model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                    nChannels=[24, 96, 384, 1536], nClasses=1000, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[3, 256, 256], mult=4)
    else:
        teacher_model = iRevNet(nBlocks=[18, 18, 18], nStrides=[1, 2, 2],
                        nChannels=[16, 64, 256], nClasses=10, init_ds=2,
                        dropout_rate=0., affineBN=True, in_shape=[3, 32, 32],
                        mult=4)
    
    return model, teacher_model