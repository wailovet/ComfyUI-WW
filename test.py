import numpy as np


text = """(masterpiece, best quality:1.2), (Maine Coon), cat, sketch
(masterpiece, best quality:1.2), (Bengal Cat), cat, sketch
(masterpiece, best quality:1.2), (American Wirehair), cat, sketch
(masterpiece, best quality:1.2), (Persian Cat), cat, sketch
(masterpiece, best quality:1.2), (British Shorthair), cat, sketch
(masterpiece, best quality:1.2), (Thai Siamese), cat, sketch
(masterpiece, best quality:1.2), (Peterbald), cat, sketch
(masterpiece, best quality:1.2), (Ragdoll), cat, sketch
(masterpiece, best quality:1.2), (Munchkin Cat), cat, sketch
(masterpiece, best quality:1.2), (Norwegian Forest Cat), cat, sketch
(masterpiece, best quality:1.2), (Scottish Fold), cat, sketch
(masterpiece, best quality:1.2), (Siberian Cat), cat, sketch
(masterpiece, best quality:1.2), (American Shorthair), cat, sketch
(masterpiece, best quality:1.2), (Exotic Shorthair), cat, sketch
(masterpiece, best quality:1.2), (Turkish Angora), cat, sketch
(masterpiece, best quality:1.2), (Russian Blue), cat, sketch
(masterpiece, best quality:1.2), (Savannah Cat), cat, sketch
(masterpiece, best quality:1.2), (Abyssinian), cat, sketch
(masterpiece, best quality:1.2), (Oriental Shorthair), cat, sketch
(masterpiece, best quality:1.2), (Egyptian Mau), cat, sketch
(masterpiece, best quality:1.2), (Birman), cat, sketch
(masterpiece, best quality:1.2), (Himalayan), cat, sketch
(masterpiece, best quality:1.2), (Lykoi), cat, sketch
(masterpiece, best quality:1.2), (Chausie), cat, sketch
(masterpiece, best quality:1.2), (LaPerm), cat, sketch
(masterpiece, best quality:1.2), (Japanese Bobtail), cat, sketch
(masterpiece, best quality:1.2), (Burmese), cat, sketch
(masterpiece, best quality:1.2), (Sphynx), cat, sketch
(masterpiece, best quality:1.2), (Turkish Van), cat, sketch
(masterpiece, best quality:1.2), (Devon Rex), cat, sketch
(masterpiece, best quality:1.2), (Domestic Shorthair), cat, sketch
(masterpiece, best quality:1.2), (Toyger), cat, sketch
(masterpiece, best quality:1.2), (Chartreux), cat, sketch
(masterpiece, best quality:1.2), (Somali Cat), cat, sketch
(masterpiece, best quality:1.2), (Manx), cat, sketch
(masterpiece, best quality:1.2), (European Shorthair), cat, sketch
(masterpiece, best quality:1.2), (Korat), cat, sketch
(masterpiece, best quality:1.2), (Selkirk Rex), cat, sketch
(masterpiece, best quality:1.2), (Highlander Cat), cat, sketch
(masterpiece, best quality:1.2), (Ocicat), cat, sketch
(masterpiece, best quality:1.2), (Cornish Rex), cat, sketch
(masterpiece, best quality:1.2), (American Curl), cat, sketch
(masterpiece, best quality:1.2), (Small Wild Cats), cat, sketch
(masterpiece, best quality:1.2), (Big Wild Cats), cat, sketch
(masterpiece, best quality:1.2), (American Bobtail), cat, sketch
(masterpiece, best quality:1.2), (Ragamuffin), cat, sketch
(masterpiece, best quality:1.2), (Balinese), cat, sketch
(masterpiece, best quality:1.2), (Bombay), cat, sketch"""




        # 换行分割
lines = text.split("\n")
        # 随机选择一行
line = lines[np.random.randint(0, len(lines))]


print(line)