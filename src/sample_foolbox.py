def main():
    print("run")
    import torch
    import torchvision
    import foolbox as fb
    model = torchvision.models.resnet18(pretrained=True)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    bounds = (0, 1)
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)

    print(fb.utils.accuracy(fmodel, images, labels))

    attack = fb.attacks.LinfDeepFoolAttack()

    raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)

    print("RUNED")

if __name__ == "__main__":
    main()