

def load_dataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        texts = []
        contador = 0
        if contador != 10:
            for line in lines:
                texts.append(line)
                contador += 1
    return texts

texts = load_dataset('E:\Master\TFM\Datos MIMIC-SRC\labevents.csv')
print(texts)
