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

    # ZALECENIA
    if (dostepnosc_konsultacji <= 2):
        zalecenie = zalecenie + "\n- Poprawić komunikację i zasady oceniania"

    if (punktualnosc <= 2):
        zalecenie = zalecenie + "\n- Zwrócić uwagę na punktualność"

    if (atrakcyjnosc_zajec <= 2):
        zalecenie = zalecenie + "\n- Zwiększyć atrakcyjność zajęć"
    
    if (kontakt <= 2):
        zalecenie = zalecenie + "\n- Poprawić kontakt z studentami"

    if (przygotowanie <= 2):
        zalecenie =zalecenie + "\n- Zwiększyć przygotowanie do zajęć"

    if (jasnosc_tlumaczenia <= 2):
        zalecenie = zalecenie + "\n- Poprawić jasność tłumaczenia"

    if (sposob_oceniania <= 2):
        zalecenie = zalecenie + "\n- Zrewidować sposób oceniania"
    
    return ocena, zalecenie


# DANE TESTOWE 
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

print("Ocena pierwszego prowadzącego:", wynik1[0])
if wynik1[1] != "":
    print("Zalecenie:", wynik1[1])

print("\n---\n")

print("Ocena drugiego prowadzącego:", wynik2[0])
if wynik2[1] != "":
    print("Zalecenie:", wynik2[1])

print("\n---\n")

# DODANIE MOŻLIWOŚCI WPROWADZENIA DANYCH PRZEZ UŻYTKOWNIKA
print("Czy chcesz wprowadzić dane dla trzeciego prowadzącego? (tak/nie)")
odpowiedz = input("> ")
if odpowiedz == "tak":
    print("Wprowadź dane dla trzeciego prowadzącego:")
    jasnosc_tlumaczenia = 0
    while jasnosc_tlumaczenia < 1 or jasnosc_tlumaczenia > 5:
        try:
            jasnosc_tlumaczenia = int(input("Jasność tłumaczenia (1-5): "))
        except ValueError:
            jasnosc_tlumaczenia = 0 
        
    przygotowanie = 0
    while przygotowanie < 1 or przygotowanie > 5:
        try:
            przygotowanie = int(input("Przygotowanie (1-5): "))
        except ValueError:
            przygotowanie = 0

    punktualnosc = 0
    while punktualnosc < 1 or punktualnosc > 5:        
        try:
            punktualnosc = int(input("Punktualność (1-5): "))
        except ValueError:
            punktualnosc = 0

    kontakt = 0
    while kontakt < 1 or kontakt > 5:
        try:
            kontakt = int(input("Kontakt (1-5): "))
        except ValueError:
            kontakt = 0

    sposob_oceniania = 0
    while sposob_oceniania < 1 or sposob_oceniania > 5:
        try:
            sposob_oceniania = int(input("Sposób oceniania (1-5): "))
        except ValueError:
            sposob_oceniania = 0

    dostepnosc_konsultacji = 0
    while dostepnosc_konsultacji < 1 or dostepnosc_konsultacji > 5:
        try:
            dostepnosc_konsultacji = int(input("Dostępność konsultacji (1-5): "))
        except ValueError:
            dostepnosc_konsultacji = 0

    atrakcyjnosc_zajec = 0
    while atrakcyjnosc_zajec < 1 or atrakcyjnosc_zajec > 5:
        try:
            atrakcyjnosc_zajec = int(input("Atrakcyjność zajęć (1-5): "))
        except ValueError:
            atrakcyjnosc_zajec = 0

    wynik3 = ocen_prowadzacego(
        jasnosc_tlumaczenia=jasnosc_tlumaczenia,
        przygotowanie=przygotowanie,
        punktualnosc=punktualnosc,
        kontakt=kontakt,
        sposob_oceniania=sposob_oceniania,
        dostepnosc_konsultacji=dostepnosc_konsultacji,
        atrakcyjnosc_zajec=atrakcyjnosc_zajec
    )
    print("\n---\nOcena trzeciego prowadzącego:", wynik3[0])
    if wynik3[1] != "":
        print("Zalecenie:", wynik3[1])

print("\n---\n\nKoniec programu.\n")