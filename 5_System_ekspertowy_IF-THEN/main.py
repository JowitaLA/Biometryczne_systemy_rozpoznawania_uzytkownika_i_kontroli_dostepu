def ocen_prowadzacego(
    jasnosc_tlumaczenia,
    przygotowanie,
    punktualnosc,
    kontakt,
    sposob_oceniania,
    dostepnosc_konsultacji,
    atrakcyjnosc_zajec
):
    ocena = ""
    zalecenie = ""

    # DODANIE REGUŁ OCENIANIA
    # wzorowa
    if (jasnosc_tlumaczenia >= 5 and przygotowanie >= 5 and kontakt >= 5 and punktualnosc >= 5 and sposob_oceniania >= 5 and dostepnosc_konsultacji >= 5 and atrakcyjnosc_zajec >= 5):
        ocena = "wzorowa"

    # bardzo dobra
    elif (jasnosc_tlumaczenia >= 4 and przygotowanie >= 4 and kontakt >= 4):
        ocena = "bardzo dobra"
    
    elif (dostepnosc_konsultacji >= 4 and atrakcyjnosc_zajec >= 4):
        ocena = "bardzo dobra"

    # dobra
    elif (punktualnosc >= 4 and przygotowanie >= 4 and sposob_oceniania >= 4):
        ocena = "dobra"

    # średnia / przeciętna
    elif (jasnosc_tlumaczenia == 3 and kontakt == 3 and atrakcyjnosc_zajec == 3):
        ocena = "przeciętna"

    # słaba
    elif (jasnosc_tlumaczenia <= 2 and kontakt <= 2):
        ocena = "słaba"

    elif (punktualnosc <= 2 and przygotowanie <= 2):
        ocena = "słaba"

    # bardzo słaba
    elif (przygotowanie <= 1 and punktualnosc <= 1):
        ocena = "bardzo słaba"

    elif (kontakt <= 1 and przygotowanie <= 1):
        ocena = "bardzo słaba"

    else:
        ocena = "przeciętna"

    # --- ZALECENIA ---
    if (dostepnosc_konsultacji <= 2):
        zalecenie = zalecenie + "Poprawić komunikację i zasady oceniania\n"

    elif (punktualnosc <= 2):
        zalecenie = zalecenie + "Zwrócić uwagę na punktualność\n"

    elif (atrakcyjnosc_zajec <= 2):
        zalecenie = zalecenie + "Zwiększyć atrakcyjność zajęć\n"
    
    elif (kontakt <= 2):
        zalecenie = zalecenie + "Poprawić kontakt z studentami\n"

    elif (przygotowanie <= 2):
        zalecenie =zalecenie + "Zwiększyć przygotowanie do zajęć\n"

    elif (jasnosc_tlumaczenia <= 2):
        zalecenie = zalecenie + "Poprawić jasność tłumaczenia\n"

    elif (sposob_oceniania <= 2):
        zalecenie = zalecenie + "Zrewidować sposób oceniania\n"
    
    return ocena, zalecenie


# DANE TESTOWE #
wynik1 = ocen_prowadzacego(
    jasnosc_tlumaczenia=5,
    przygotowanie=4,
    punktualnosc=5,
    kontakt=4,
    sposob_oceniania=4,
    dostepnosc_konsultacji=4,
    atrakcyjnosc_zajec=5
)

wynik2 = ocen_prowadzacego(
    jasnosc_tlumaczenia=2,
    przygotowanie=3,
    punktualnosc=2,
    kontakt=3,
    sposob_oceniania=3,
    dostepnosc_konsultacji=2,
    atrakcyjnosc_zajec=5
)

print("Ocena prowadzącego:", wynik1[0])
if wynik1[1] != "":
    print("Zalecenie:", wynik1[1])

print("\n---\n")

print("Ocena prowadzącego:", wynik2[0])
if wynik2[1] != "":
    print("Zalecenie:", wynik2[1])