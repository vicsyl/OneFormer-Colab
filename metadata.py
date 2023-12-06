# METADATA: Metadata(evaluator_type='ade20k_panoptic_seg',
#                    ignore_label=255,
#                    image_root='datasets/ADEChallengeData2016/images/validation',
#                    json_file='datasets/ADEChallengeData2016/ade20k_instance_val.json',
#                    label_divisor=1000,
#                    name='ade20k_panoptic_val',
#                    panoptic_json='datasets/ADEChallengeData2016/ade20k_panoptic_val.json',
#                    panoptic_root='datasets/ADEChallengeData2016/ade20k_panoptic_val',

thing_colors = [[204, 5, 255], [230, 230, 230], [224, 5, 255], [150, 5, 61], [8, 255, 51],
                [255, 6, 82], [255, 51, 7], [204, 70, 3], [0, 102, 200], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [220, 220, 220], [8, 255, 214], [7, 255, 224], [255, 184, 6],
                [10, 255, 71], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6],
                [255, 194, 7], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255],
                [0, 163, 255], [250, 10, 15], [20, 255, 0], [255, 224, 0], [0, 0, 255], [255, 71, 0],
                [0, 235, 255], [0, 173, 255], [0, 255, 245], [0, 255, 112], [0, 255, 133], [255, 0, 0],
                [255, 163, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41],
                [0, 255, 173], [10, 0, 255], [173, 255, 0], [255, 92, 0], [255, 0, 245], [255, 0, 102],
                [255, 173, 0], [255, 0, 20], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204],
                [0, 255, 194], [0, 255, 82], [0, 112, 255], [51, 0, 255], [0, 122, 255], [255, 153, 0],
                [0, 255, 10], [163, 255, 0], [255, 235, 0], [8, 184, 170], [184, 0, 255], [255, 0, 31],
                [0, 214, 255], [255, 0, 112], [92, 255, 0], [70, 184, 160], [163, 0, 255],
                [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0], [214, 255, 0],
                [0, 204, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204], [41, 0, 255],
                [41, 255, 0], [173, 0, 255], [0, 245, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
                [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]]

stuff_colors = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
                [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
                [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230],
                [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
                [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41],
                [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
                [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255],
                [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
                [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255],
                [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0],
                [0, 255, 153], [255, 92, 0], [255, 0, 255], [255, 0, 245], [255, 0, 102],
                [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255],
                [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10],
                [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
                [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
                [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
                [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
                [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255], [71, 0, 255], [122, 0, 255],
                [0, 255, 184], [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0],
                [25, 194, 194], [102, 255, 0], [92, 0, 255]]


stuff_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road, route', 'bed',
                 'window ', 'grass', 'cabinet', 'sidewalk, pavement', 'person', 'earth, ground',
                 'door', 'table', 'mountain, mount', 'plant', 'curtain', 'chair', 'car', 'water',
                 'painting, picture', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
                 'armchair', 'seat', 'fence', 'desk', 'rock, stone', 'wardrobe, closet, press', 'lamp',
                 'tub', 'rail', 'cushion', 'base, pedestal, stand', 'box', 'column, pillar',
                 'signboard, sign', 'chest of drawers, chest, bureau, dresser', 'counter', 'sand',
                 'sink', 'skyscraper', 'fireplace', 'refrigerator, icebox',
                 'grandstand, covered stand', 'path', 'stairs', 'runway',
                 'case, display case, showcase, vitrine', 'pool table, billiard table, snooker table',
                 'pillow', 'screen door, screen', 'stairway, staircase', 'river', 'bridge, span',
                 'bookcase', 'blind, screen', 'coffee table',
                 'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm, palm tree', 'kitchen island', 'computer',
                 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel, hut, hutch, shack, shanty',
                 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning, sunshade, sunblind',
                 'street lamp', 'booth', 'tv', 'plane', 'dirt track', 'clothes', 'pole',
                 'land, ground, soil', 'bannister, banister, balustrade, balusters, handrail',
                 'escalator, moving staircase, moving stairway',
                 'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'buffet, counter, sideboard',
                 'poster, posting, placard, notice, bill, card', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy',
                 'washer, automatic washer, washing machine', 'plaything, toy', 'pool', 'stool',
                 'barrel, cask', 'basket, handbasket', 'falls', 'tent', 'bag', 'minibike, motorbike',
                 'cradle', 'oven', 'ball', 'food, solid food', 'step, stair', 'tank, storage tank',
                 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen',
                 'blanket, cover', 'sculpture', 'hood, exhaust hood', 'sconce', 'vase',
                 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
                 'bulletin board', 'shower', 'radiator', 'glass, drinking glass', 'clock', 'flag']

stuff_dataset_id_to_contiguous_id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                                     10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
                                     18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25,
                                     26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33,
                                     34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41,
                                     42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49,
                                     50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57,
                                     58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63, 64: 64, 65: 65,
                                     66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71, 72: 72, 73: 73,
                                     74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80, 81: 81,
                                     82: 82, 83: 83, 84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89,
                                     90: 90, 91: 91, 92: 92, 93: 93, 94: 94, 95: 95, 96: 96, 97: 97,
                                     98: 98, 99: 99, 100: 100, 101: 101, 102: 102, 103: 103, 104: 104,
                                     105: 105, 106: 106, 107: 107, 108: 108, 109: 109, 110: 110,
                                     111: 111, 112: 112, 113: 113, 114: 114, 115: 115, 116: 116,
                                     117: 117, 118: 118, 119: 119, 120: 120, 121: 121, 122: 122,
                                     123: 123, 124: 124, 125: 125, 126: 126, 127: 127, 128: 128,
                                     129: 129, 130: 130, 131: 131, 132: 132, 133: 133, 134: 134,
                                     135: 135, 136: 136, 137: 137, 138: 138, 139: 139, 140: 140,
                                     141: 141, 142: 142, 143: 143, 144: 144, 145: 145, 146: 146,
                                     147: 147, 148: 148, 149: 149}

thing_classes = ['bed', 'window ', 'cabinet', 'person', 'door', 'table', 'curtain', 'chair', 'car',
                 'painting, picture', 'sofa', 'shelf', 'mirror', 'armchair', 'seat', 'fence', 'desk',
                 'wardrobe, closet, press', 'lamp', 'tub', 'rail', 'cushion', 'box', 'column, pillar',
                 'signboard, sign', 'chest of drawers, chest, bureau, dresser', 'counter', 'sink',
                 'fireplace', 'refrigerator, icebox', 'stairs',
                 'case, display case, showcase, vitrine', 'pool table, billiard table, snooker table',
                 'pillow', 'screen door, screen', 'bookcase', 'coffee table',
                 'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower', 'book', 'bench',
                 'countertop', 'stove', 'palm, palm tree', 'kitchen island', 'computer',
                 'swivel chair', 'boat', 'arcade machine', 'bus', 'towel', 'light', 'truck',
                 'chandelier', 'awning, sunshade, sunblind', 'street lamp', 'booth', 'tv', 'plane',
                 'clothes', 'pole', 'bannister, banister, balustrade, balusters, handrail',
                 'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'van', 'ship', 'fountain',
                 'washer, automatic washer, washing machine', 'plaything, toy', 'stool',
                 'barrel, cask', 'basket, handbasket', 'bag', 'minibike, motorbike', 'oven', 'ball',
                 'food, solid food', 'step, stair', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'dishwasher', 'screen', 'sculpture', 'hood, exhaust hood', 'sconce',
                 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'plate', 'monitor',
                 'bulletin board', 'radiator', 'glass, drinking glass', 'clock', 'flag']

thing_dataset_id_to_contiguous_id = {7: 7, 8: 8, 10: 10, 12: 12, 14: 14, 15: 15, 18: 18, 19: 19,
                                     20: 20, 22: 22, 23: 23, 24: 24, 27: 27, 30: 30, 31: 31, 32: 32,
                                     33: 33, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 41: 41, 42: 42,
                                     43: 43, 44: 44, 45: 45, 47: 47, 49: 49, 50: 50, 53: 53, 55: 55,
                                     56: 56, 57: 57, 58: 58, 62: 62, 64: 64, 65: 65, 66: 66, 67: 67,
                                     69: 69, 70: 70, 71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76,
                                     78: 78, 80: 80, 81: 81, 82: 82, 83: 83, 85: 85, 86: 86, 87: 87,
                                     88: 88, 89: 89, 90: 90, 92: 92, 93: 93, 95: 95, 97: 97, 98: 98,
                                     102: 102, 103: 103, 104: 104, 107: 107, 108: 108, 110: 110,
                                     111: 111, 112: 112, 115: 115, 116: 116, 118: 118, 119: 119,
                                     120: 120, 121: 121, 123: 123, 124: 124, 125: 125, 126: 126,
                                     127: 127, 129: 129, 130: 130, 132: 132, 133: 133, 134: 134,
                                     135: 135, 136: 136, 137: 137, 138: 138, 139: 139, 142: 142,
                                     143: 143, 144: 144, 146: 146, 147: 147, 148: 148, 149: 149}

arkit_class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
    "sink", "washer", "toilet", "bathtub", "oven", # 5..10
    "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
    "tv_monitor", "sofa", # 15..17
]

arkit_class_names_map = {
    "cabinet": set(["cabinet", "chest of drawers, chest, bureau, dresser"]),
    "refrigerator": set(["refrigerator, icebox"]),
    "shelf": set(["shelf"]),
    "stove": set(["stove"]),
    "bed": set(["bed"]),
    "sink": set(["sink"]),
    "washer": set(["washer, automatic washer, washing machine"]),
    "toilet": set(["toilet, can, commode, crapper, pot, potty, stool, throne"]),
    "bathtub": set(["tub"]),
    "oven": set(["oven"]),
    "dishwasher": set(["dishwasher"]),
    "fireplace": set(["fireplace"]),
    # see toilet
    "stool": set(["stool"]),
    "chair": set(["chair", "armchair", "swivel chair", "seat"]),
    "table": set(["pool table, billiard table, snooker table", "coffee table", "table", "desk"]),
    "tv_monitor": set(["monitor", "crt screen", "computer"]),
    "sofa": set(["sofa"]),
}


def get_reverse_map():
    ret = {}
    for arkit_c, sem_classes in arkit_class_names_map.items():
        for sem_class in sem_classes:
            ret[sem_class] = arkit_c
    return ret


reverse_arkit_class_names_map = get_reverse_map()


def test_arkit_class_names_map():

    assert len(arkit_class_names_map) == len(arkit_class_names)
    already_present = set()
    for k,vs in arkit_class_names_map.items():
        assert k in arkit_class_names
        for v in vs:
            assert not already_present.__contains__(v)
            already_present.add(v)
            assert v in stuff_classes


if __name__ == "__main__":
    test_arkit_class_names_map()
