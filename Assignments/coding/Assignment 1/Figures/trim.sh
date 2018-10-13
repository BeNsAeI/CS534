#!/bin/bash
#for i in {1..2893}
for i in {1..7} 
do
	convert CS534-1-${i}.png -crop 1400x700+100+100 CS534-1-${i}.png
	echo frame${i}.jpg resized
done

