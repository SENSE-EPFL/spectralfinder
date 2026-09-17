from hapi import *

db_begin('data')  # crée un dossier local pour stocker les données

# Télécharge les raies pour une molécule sur une plage spectrale (en cm^-1)
#fetch('H2O', 1, 1, 3400, 3700)  # (nom_table, molecule_id, isotopologue_id, nu_min, nu_max)

# Calcule un spectre d'absorption
nu, coef = absorptionCoefficient_Voigt(SourceTables='H2O')
print("l'abondance de H2O :", abundance(1, 2))  # Affiche l'abondance de la molécule

import matplotlib.pyplot as plt

plt.plot(nu, coef)
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylabel('Absorption coefficient (cm$^{-1}$)')
plt.title('H2O absorption spectrum')
plt.show()