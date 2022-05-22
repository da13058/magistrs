# Dmitrijs Arkašarins, da16019
# UZ ATVĒRTIEM DATIEM BALSTĪTA REĢIONAM LĪDZĪGĀ RELJEFA ĢENERĒŠANA, maģistra darbs

file_object  = open("OUTPUT_GENERIC.file.txt", "r")

mylist = list()

minX = 0
minY = 0
maxX = 0
maxY = 0
maxHeight = 0

heightA = 0
heightB = 0
heightC = 0
heightD = 0

while True:
    a = file_object.readline().split()
    if a is None or len(a) == 0 or a is EOFError:
        break
    else:
        if (minX == 0) or (minX > a[0]):
           minX = a[0]
        if (minY == 0 or minY > a[1]):
           minY = a[1]
        if (maxX == 0 or maxX < a[0]):
           maxX = a[0]
        if (maxY == 0 or maxY < a[1]):
           maxY = a[1]
        if (maxHeight == 0 or maxHeight < a[2]):
           maxHeight = a[2]
        if (minX == a[0]) and (minY == a[1]):
            heightA = a[2]
        if (minX == a[0]) and (maxY == a[1]):
            heightB = a[2]
        if (maxX == a[0]) and (maxY == a[1]):
            heightC = a[2]
        if (maxX == a[0]) and (minY == a[1]):
            heightD = a[2]
        mylist.append(a)

print(minX)       
print(minY)       

print(maxX)       

print(maxY)
print(maxHeight)

print(heightA)
print(heightB)
print(heightC)
print(heightD)


