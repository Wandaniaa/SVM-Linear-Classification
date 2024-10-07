from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import Person
import pandas as pd
from .resources import PersonResource
from django.core.files.storage import FileSystemStorage
from .forms import EditPersonForm
from .forms import AddPersonForm
from django.urls import reverse
from .models import normalize
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
from .utils import klasifikasi_berhak
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging

# Create your views here.
def BASE(request):
    return render(request, 'base.html')

def index2_view(request):
    return render(request, 'index2.html')

def base_view(request):
    return render(request, 'base.html')

def advanced_view(request):
    persons = Person.objects.all()
 #Fungsi-fungsi normalisasi
def normalize_pendapatan(value):
    # Implementasikan logika normalisasi bobot_pendapatan
    return normalize(value, 1, 3, 0, 1)

def normalize_pt(value):
    # Implementasi logika normalisasi bobot_pt
    # ...
    return normalize(value, 1, 2, 0, 1)

def normalize_pengalaman(value):
    # Implementasi logika normalisasi bobot_pengalaman
    # ...
    return normalize(value, 1, 2, 0, 1)

def normalize_status(value):
    # Implementasi logika normalisasi bobot_status
    # ...
    return normalize(value, 1, 4, 0, 1)

# Fungsi untuk menampilkan data yang telah dinormalisasi
def advanced_view(request):
    persons = Person.objects.all()
    
    # Lakukan normalisasi pada setiap objek Person
    for person in persons:
        # Implementasikan normalisasi di sini
        person.bobot_pendapatan_normalized = normalize_pendapatan(person.bobot_pendapatan)
        person.bobot_pt_normalized = normalize_pt(person.bobot_pt)
        person.bobot_pengalaman_normalized = normalize_pengalaman(person.bobot_pengalaman)
        person.bobot_status_normalized = normalize_status(person.bobot_status)

    return render(request, 'advanced.html', {'persons': persons})

def get_next_person_number():
    last_person = Person.objects.order_by('-no').first()
    if last_person:
        next_number = last_person.no + 1
    else:
        next_number = 1
    print("Next Person Number:", next_number)
    return next_number
logger = logging.getLogger(__name__)

def add_person(request):
    logger.info("Entering add_person function")

    if request.method == 'POST':
        form = AddPersonForm(request.POST)
        if form.is_valid():
            person = form.save(commit=False)

            # Cetak nilai sebelum pengaturan person.no
            print("Before setting person.no:", get_next_person_number())

            person.no = get_next_person_number()  # Pastikan bidang no diatur

            # Cetak nilai setelah pengaturan person.no
            print("After setting person.no:", get_next_person_number())

            hasil_klasifikasi = klasifikasi_berhak(person)
            person.hasil_klasifikasi = hasil_klasifikasi
            person.save()
            logger.info(f"Person {person.nama} classified as {hasil_klasifikasi}")

            # Menambahkan hasil klasifikasi ke dalam konteks
            context = {'form': form, 'hasil_klasifikasi': hasil_klasifikasi}
            return render(request, 'add_person.html', context)
        else:
            logger.error(f"Form is not valid: {form.errors}")
    else:
        form = AddPersonForm()

    logger.info("Rendering add_person.html")
    return render(request, 'add_person.html', {'form': form})

def edit_person(request, pk):
    person = get_object_or_404(Person, pk=pk)
    if request.method == 'POST':
        form = EditPersonForm(request.POST, instance=person)
        if form.is_valid():
            form.save()
            return redirect(reverse('advanced'))  # Menggunakan reverse untuk mendapatkan URL
    else:
        form = EditPersonForm(instance=person)

    return render(request, 'edit_person.html', {'form': form})

def delete_person(request, pk):
    person = get_object_or_404(Person, pk=pk)
    if request.method == 'POST':
        person.delete()
        return redirect('advanced')  # Redirect ke halaman data lanjutan setelah menghapus data
    else:
        return render(request, 'delete_person.html', {'person': person})

def normalisasi_view(request):
    person = Person.objects.all



def general_view(request):
    success_message = None

    if request.method == 'POST':
        try:
            uji_coba_file = request.FILES['uji_coba']

            # Simpan file ke sistem penyimpanan
            fs = FileSystemStorage()
            filename = fs.save(uji_coba_file.name, uji_coba_file)

            # Baca data Excel menggunakan Pandas
            empexceldata = pd.read_excel(fs.open(filename))

            # Cetak nama kolom
            print(empexceldata.columns)

            # Iterasi melalui baris DataFrame dan tambahkan ke database
            for _, row in empexceldata.iterrows():
                obj = Person.objects.create(
                    no=row['NO'],
                    nama=row['NAMA'],
                    nik=row['NIK'],
                    kecamatan=row['KECAMATAN'],
                    desa=row['DESA'],
                    pendapatan=row['pendapatan'],
                    bobot_pendapatan=row['bobot_pendapatan'],
                    profesi_utama=row['profesi_utama'],
                    profesi_tambahan=row['profesi_tambahan'],
                    bobot_pt=row['bobot_pt'],
                    pengalaman=row['pengalaman'],
                    bobot_pengalaman=row['bobot_pengalaman'],
                    status=row['status'],
                    bobot_status=row['bobot_status'],
                    label=row['LABEL']
                )
                obj.save()

            # Hapus file yang diunggah setelah impor selesai
            fs.delete(filename)

            success_message = "File berhasil diimpor."

            # Redirect ke halaman advanced.html setelah impor berhasil
            return redirect('advanced')

        except Exception as e:
            # Tangani kesalahan lainnya
            print(f"Error: {str(e)}")
            return HttpResponse(f"Terjadi kesalahan: {str(e)}")

    return render(request, 'general.html', {'success_message': success_message})

def evaluate_classification(data_true, data_pred):
    # Menghitung confusion matrix
    cm = confusion_matrix(data_true, data_pred)

    # Menghitung precision dan recall
    precision = precision_score(data_true, data_pred, average=None)
    recall = recall_score(data_true, data_pred, average=None)

    return cm, precision, recall

def classify_svm_linear(data_latih, data_uji):
    # Memisahkan fitur dan label untuk data latih
    X_train = [[person.bobot_pendapatan_normalized, person.bobot_pt_normalized, person.bobot_pengalaman_normalized, person.bobot_status_normalized] for person in data_latih]
    y_train = [person.label for person in data_latih]

    # Membuat dan melatih model SVM linear
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Melakukan prediksi pada data latih
    X_train = [[person.bobot_pendapatan_normalized, person.bobot_pt_normalized, person.bobot_pengalaman_normalized, person.bobot_status_normalized] for person in data_latih]
    y_train_true = [person.label for person in data_latih]
    y_train_pred = clf.predict(X_train)

    for person, predicted_label in zip(data_latih, y_train_pred):
        person.predicted_label = predicted_label
        person.save()

    # Melakukan prediksi pada data uji
    X_test = [[person.bobot_pendapatan_normalized, person.bobot_pt_normalized, person.bobot_pengalaman_normalized, person.bobot_status_normalized] for person in data_uji]
    y_true = [person.label for person in data_uji]
    y_pred = clf.predict(X_test)

    for person, predicted_label in zip(data_uji, y_pred):
        person.predicted_label = predicted_label
        person.save()

    # Evaluasi model menggunakan akurasi
    accuracy_train = accuracy_score(y_train_true, y_train_pred)
    accuracy_test = accuracy_score(y_true, y_pred)

    cm_train, precision_train, recall_train = evaluate_classification(y_train_true, y_train_pred)
    cm_test, precision_test, recall_test = evaluate_classification(y_true, y_pred)

    return accuracy_train, accuracy_test, cm_train, precision_train, recall_train, cm_test, precision_test, recall_test

def klasifikasi_view(request):
    persons = Person.objects.all()

    # Normalisasi data
    for person in persons:
        person.save()

    # Memisahkan data menjadi data latih dan data uji
    data_latih = persons[:320]
    data_uji = persons[320:400]

    # Melakukan klasifikasi SVM linear dan mendapatkan akurasi
    accuracy_train, accuracy_test, cm_train, precision_train, recall_train, cm_test, precision_test, recall_test = classify_svm_linear(data_latih, data_uji)

    return render(request, 'klasifikasi.html', {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test, 'persons': persons, 'data_latih': data_latih, 'data_uji': data_uji})

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Simpan plot ke BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Ubah gambar ke dalam format yang dapat ditampilkan di HTML
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode("utf-8")
    plt.close()

    return graphic

def plot_precision_recall_curve(precision, recall, title):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', alpha=0.2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save plot to BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Convert image to Base64 for HTML display
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode("utf-8")
    plt.close()

    return graphic

def evaluasi_view(request):
    persons = Person.objects.all()

    # Normalisasi data
    for person in persons:
        person.save()

    # Memisahkan data menjadi data latih dan data uji
    data_latih = persons[:320]
    data_uji = persons[320:400]

    # Melakukan klasifikasi SVM linear dan mendapatkan akurasi serta evaluasi tambahan
    accuracy_train, accuracy_test, _, precision_train, recall_train, _, precision_test, recall_test = classify_svm_linear(data_latih, data_uji)

    # Cetak nilai recall untuk memeriksa
    print("Recall Train:", recall_train)
    print("Recall Test:", recall_test)

    # Menghitung confusion matrix untuk data latih
    y_train_true = [person.label for person in data_latih]
    y_train_pred = [person.predicted_label for person in data_latih]
    cm_train = confusion_matrix(y_train_true, y_train_pred)

    # Label kelas
    labels = ["Class 0", "Class 1"]  # Sesuaikan dengan label kelas Anda

    # Membuat grafik dari confusion matrix untuk data latih
    cm_train_plot = plot_confusion_matrix(cm_train, labels)

    # Menghitung confusion matrix untuk data uji
    _, _, cm_test, _, _, _, _, _ = classify_svm_linear(data_latih, data_uji)

    # Membuat grafik dari confusion matrix untuk data uji
    cm_test_plot = plot_confusion_matrix(cm_test, labels)

    # Plot grafik presisi untuk data latih
    precision_train_plot = plot_precision_recall_curve(precision_train, recall_train, 'Precision-Recall Curve - Data Latih')

    # Plot grafik presisi untuk data uji
    precision_test_plot = plot_precision_recall_curve(precision_test, recall_test, 'Precision-Recall Curve - Data Uji')

# Plot grafik recall untuk data latih
    plt.figure(figsize=(8, 6))
    plt.plot(recall_train, label='Recall Curve - Data Latih')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall Curve - Data Latih')
    plt.legend()
    plt.tight_layout()

    # Simpan plot ke BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Ubah gambar ke dalam format yang dapat ditampilkan di HTML
    recall_train_plot = base64.b64encode(image_png).decode("utf-8")
    plt.close()

    # Plot grafik recall untuk data uji
    plt.figure(figsize=(8, 6))
    plt.plot(recall_test, label='Recall Curve - Data Uji')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall Curve - Data Uji')
    plt.legend()
    plt.tight_layout()

    # Simpan plot ke BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Ubah gambar ke dalam format yang dapat ditampilkan di HTML
    recall_test_plot = base64.b64encode(image_png).decode("utf-8")
    plt.close()

    return render(request, 'evaluasi.html', {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'cm_train': cm_train_plot,
        'cm_test': cm_test_plot,
        'precision_train': precision_train,
        'recall_train': recall_train,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'precision_train_plot': precision_train_plot,
        'precision_test_plot': precision_test_plot,
        'recall_train_plot': recall_train_plot,
        'recall_test_plot': recall_test_plot,
        'persons': persons,
        'data_latih': data_latih,
        'data_uji': data_uji
    })







