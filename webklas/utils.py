def klasifikasi_berhak(person):
    # Logika klasifikasi
    if (
        person.bobot_pendapatan <= 1 or
        person.bobot_pt <= 1 or
        person.bobot_pengalaman <= 1 or
        person.bobot_status <= 1
    ):
        return 'Tidak Berhak'
    else:
        return 'Berhak'