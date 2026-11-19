"""Modul za predobdelavo, filtriranje in čiščenje oblakov točk (2. korak).

Predvidene funkcionalnosti v 2. koraku:
- prostorsko obrezovanje (cropping / ROI),
- odstranjevanje osamelcev (statistično filtriranje s k-NN, radijalno filtriranje),
- podvzorčenje (voxel grid / naključno vzorčenje),
- ocenjevanje normal in segmentacija geometrijskih primitivov.
"""
