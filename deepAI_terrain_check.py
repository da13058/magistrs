# Dmitrijs Arkašarins, da16019
# UZ ATVĒRTIEM DATIEM BALSTĪTA REĢIONAM LĪDZĪGĀ RELJEFA ĢENERĒŠANA, maģistra darbs

import requests, time
freeAPIKey = 'quickstart-QUdJIGlzIGNvbWluZy4uLi4K' # dažus testus var veikt ar sniegto bezmaksas API atslēgto, bet tālāk jāiegūst sava bezmaksas atslēga, piereģistrējoties Deep AI kopienā
userAPIKey = '71bd34fc-a7ac-*************' # privāta atslēga neierobežotai lietošanai
accessHeader = {'api-key': userAPIKey}
apiURL = 'https://api.deepai.org/api/image-similarity'

rootDirectory = 'C:/Users/dmitr/OneDrive/Documents/4411'

terrainType = 'hillshade'
lidarCoordinates = '4411-14-54'
generatedHeightmap = open(f'C:/Users/dmitr/OneDrive/Documents/4411/{terrainType}/output/train-99.png', 'rb')

def getOriginalHeightmap(terrainType, lidarCoordinates):
    return open(f'{rootDirectory}/{terrainType}/{lidarCoordinates}_{terrainType}.png', 'rb')

def callImageSimilarityAPI(generatedHeightmap, terrainType, lidarCoordinates):
    originalHeightmap = getOriginalHeightmap(terrainType, lidarCoordinates)
    print(f'Sending heightmaps to {apiURL}')
    r = requests.post(
        apiURL,
        files={
            'image1': originalHeightmap,
            'image2': generatedHeightmap
        },
        headers=accessHeader
    )
    print(f'Result for {terrainType} {lidarCoordinates} {r.json()["output"]}')
    originalHeightmap.close()

callImageSimilarityAPI(generatedHeightmap, terrainType, lidarCoordinates)







































