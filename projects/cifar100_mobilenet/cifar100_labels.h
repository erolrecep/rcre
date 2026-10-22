/**
 * @file cifar100_labels.h
 * @brief Canonical CIFAR-100 Fine & Coarse Class Label Tables
 */

#ifndef CIFAR100_LABELS_H
#define CIFAR100_LABELS_H

#define CIFAR100_NUM_CLASSES 100
#define CIFAR100_NUM_SUPERCLASSES 20

static const char* const CIFAR100_CLASSES[CIFAR100_NUM_CLASSES] = {
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm"
};

static const char* const CIFAR100_SUPERCLASSES[CIFAR100_NUM_SUPERCLASSES] = {
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man_made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_sized_mammals",
    "non_insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2"
};

/* Mapping of fine class ID (0..99) to superclass ID (0..19) */
static const int CIFAR100_FINE_TO_COARSE[CIFAR100_NUM_CLASSES] = {
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    3, 14,  9, 19,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
   16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
   10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
   18,  1,  2, 15,  6,  0, 17,  8, 14, 13
};

#endif /* CIFAR100_LABELS_H */
