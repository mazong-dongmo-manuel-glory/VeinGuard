import gpiozero
from time import sleep
ledVerte = gpiozero.LED(16)
ledRouge = gpiozero.LED(22)
print("cc")
while True:
   print("je suis allume")
   ledVerte.on()
   ledRouge.on()
   sleep(1)
   ledVerte.off()
   ledRouge.off()
   sleep(1)
