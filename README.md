# magistrs

Maģistra darba ietvaros izstrādāti skripti un datnes.

1) lastosquare.py – skripts, kas iegūst LiDAR reljefa objekta kvadrāta parametrus, jeb tā virsotņu augstumus un vislielāko augstuma vērtību;
2) deepAI_terrain_check.py – skripts, kas savienojas ar Deep AI veidoto Image Similarity API, aizsūtot reljefa paraugus un rezultātā iegūstot līdzības novērtējumu
3) LiDAR data – direktorija ar augstumu kartēm kas tika izveidotas no lidara (“full” – pilnā mākoņpunktu kopa, “hillshade” – digitālais reljefa modelis), GAN ģeneratora rezultāti kopā ar trenēšanas datni, kā arī fraktāļu modeļu augstumu kartes, izveidotas pēc lidara teksta datnes (pati datne sver 165mb, GitHub liedz tās ielādi, bet to iespējams iegūt pie autora).
3) gan_terrain_generator.py – Python skripts reljefa attēlu ģenerēšanai, izmantojot no atvērtajiem datiem iegūtas reljefa augstumu kartes un pielietojot GAN neironu tīklu tehnoloģiju
