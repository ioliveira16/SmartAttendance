import face_recognition
import cv2
import os
import sqlite3
from datetime import datetime, timedelta
import logging
from PIL import Image, ImageDraw, ImageFont
import qrcode

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Variáveis globais
DB_FILE = "school_attendance.db"
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
BADGES_DIR = "badges"
CAMERA_INDEX = 0

# Inicialização de listas e dicionários
known_face_encodings = []
known_face_names = []
recognized_faces = set()  # Armazena nomes dos rostos já reconhecidos
last_recognition_time = {}  # Dicionário para armazenar o último horário de registro por pessoa


# Configuração do banco de dados
def setup_database():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(BADGES_DIR, exist_ok=True)

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date_time TEXT NOT NULL,
            status TEXT NOT NULL,
            is_new INTEGER DEFAULT 0,
            person_type TEXT DEFAULT 'Desconhecido',
            internal_number TEXT DEFAULT 'N/A'
        )
        ''')

        logging.info("Banco de dados configurado com sucesso.")


# Registro de presença
def register_attendance(name, status, is_new=False, person_type="Desconhecido", internal_number="N/A"):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO attendance (name, date_time, status, is_new, person_type, internal_number) VALUES (?, ?, ?, ?, ?, ?)",
            (name, date_time, status, int(is_new), person_type, internal_number)
        )
        logging.info(f"Registro de presença salvo: {name}, Status: {status}")


# Carregamento de rostos conhecidos
def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()

    logging.info("Carregando rostos conhecidos...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                logging.info(f"Rosto carregado: {filename}")
            else:
                logging.warning(f"Nenhum rosto encontrado em: {filename}")

    logging.info(f"{len(known_face_names)} rostos conhecidos carregados.")


# Reconhecimento facial ao vivo com verificação de intervalo
def recognize_faces_from_camera():
    logging.info("Iniciando reconhecimento facial ao vivo...")
    video_capture = cv2.VideoCapture(CAMERA_INDEX)

    if not video_capture.isOpened():
        logging.error("Erro: Não foi possível acessar a câmera.")
        return

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Erro ao capturar a imagem da câmera.")
                break

            # Converte BGR para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")

            # Adiciona um pequeno atraso para verificar as teclas frequentemente
            key = cv2.waitKey(10)
            if key in [27, ord('q')]:  # ESC (27) ou Q
                logging.info("Saindo do modo de leitura ao vivo.")
                break

            if face_locations:
                try:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Desconhecido"

                        if True in matches:
                            match_index = matches.index(True)
                            name = known_face_names[match_index]

                            # Verificar se já passou mais de 1 minuto desde a última leitura
                            now = datetime.now()
                            if name not in last_recognition_time or (now - last_recognition_time[name] > timedelta(minutes=1)):
                                last_recognition_time[name] = now
                                register_attendance(name, "Presente")
                                logging.info(f"Rosto reconhecido: {name}")
                            else:
                                logging.info(f"Rosto de {name} já reconhecido recentemente. Ignorando.")
                        else:
                            if "Desconhecido" not in recognized_faces:
                                recognized_faces.add("Desconhecido")
                                save_unknown_face(frame, (top, right, bottom, left), face_encoding)

                        # Desenhar o retângulo ao redor do rosto
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                except Exception as e:
                    logging.error(f"Erro ao processar rostos: {e}")

            # Exibir a saída ao vivo
            cv2.imshow("Reconhecimento Facial Ao Vivo", frame)

    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        logging.info("Reconhecimento facial encerrado.")


# Salvar rosto desconhecido e gerar crachá
def save_unknown_face(frame, location, encoding):
    (top, right, bottom, left) = location
    face_image = frame[top:bottom, left:right]
    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

    choice = input("Rosto desconhecido detectado. Deseja cadastrar? (s/n): ").strip().lower()
    if choice == "s":
        name = input("Digite o nome para este rosto: ").strip()
        person_type = input("Digite o tipo de pessoa (INTERNO, EXTERNO, ALUNO, PROFESSOR): ").strip()
        internal_number = input("Digite o número interno (opcional): ").strip()

        # Salvar rosto
        save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        cv2.imwrite(save_path, face_image_bgr)
        known_face_encodings.append(encoding)
        known_face_names.append(name)

        # Registrar no banco de dados
        register_attendance(name, "Cadastrado", is_new=True, person_type=person_type, internal_number=internal_number)

        # Gerar o crachá com QR Code
        badge_path = generate_badge(name, person_type, internal_number, save_path)
        logging.info(f"Rosto desconhecido cadastrado e crachá gerado: {badge_path}")
    else:
        logging.info("Rosto desconhecido ignorado.")


# Geração de crachás com QR Code
def generate_badge(name, person_type, internal_number, face_image_path):
    badge_width = 400
    badge_height = 600
    background_color = "#f0f0f0"
    header_color = "#004aad"
    text_color = "black"
    border_color = "#004aad"

    # Criar o fundo do crachá
    badge = Image.new("RGB", (badge_width, badge_height), background_color)
    draw = ImageDraw.Draw(badge)

    # Adicionar borda
    draw.rectangle([(5, 5), (badge_width - 5, badge_height - 5)], outline=border_color, width=3)

    # Adicionar cabeçalho
    draw.rectangle([(0, 0), (badge_width, 60)], fill=header_color)

    # Fonte
    font_header = ImageFont.truetype("arial.ttf", 24)
    font_text = ImageFont.truetype("arial.ttf", 18)

    # Adicionar texto ao cabeçalho
    draw.text((10, 15), "Crachá de Identificação", fill="white", font=font_header)

    # Adicionar informações ao crachá
    draw.text((20, 80), f"Nome: {name}", fill=text_color, font=font_text)
    draw.text((20, 110), f"Tipo: {person_type}", fill=text_color, font=font_text)
    draw.text((20, 140), f"Número: {internal_number}", fill=text_color, font=font_text)

    # Adicionar QR Code
    qr_data = f"Nome: {name}\nTipo: {person_type}\nNúmero: {internal_number}"
    qr = qrcode.make(qr_data).resize((100, 100))
    badge.paste(qr, (280, 450))  # Posicionar no canto inferior direito

    # Adicionar a imagem do rosto
    if os.path.exists(face_image_path):
        face_image = Image.open(face_image_path).resize((200, 200))
        badge.paste(face_image, (100, 200))

    # Salvar o crachá
    badge_path = os.path.join(BADGES_DIR, f"{name}_badge.png")
    badge.save(badge_path)
    return badge_path


# Menu para gerar crachás manualmente
def generate_badge_menu():
    """
    Menu para gerar um crachá manualmente.
    """
    name = input("Digite o nome do colaborador para gerar o crachá: ").strip()
    person_type = input("Digite o tipo de pessoa (INTERNO, EXTERNO, ALUNO, PROFESSOR): ").strip()
    internal_number = input("Digite o número interno (opcional): ").strip()

    # Verificar se a imagem do colaborador existe
    face_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    if not os.path.exists(face_image_path):
        print(f"Imagem do colaborador '{name}' não encontrada em '{KNOWN_FACES_DIR}'.")
        return

    # Gerar o crachá
    badge_path = generate_badge(name, person_type, internal_number, face_image_path)
    print(f"Crachá gerado com sucesso: {badge_path}")


# Exibir histórico de presença
def show_attendance_history():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance ORDER BY date_time DESC")
        rows = cursor.fetchall()

        if not rows:
            logging.info("Nenhum registro encontrado.")
        else:
            print("\n===== Histórico de Presença =====")
            for row in rows:
                print(
                    f"Nome: {row[1]}\nData/Hora: {row[2]}\nStatus: {row[3]}\nTipo: {row[5]}\nNúmero Interno: {row[6]}\n{'-'*40}"
                )


# Menu principal
def main_menu():
    while True:
        print("\n===== Menu Principal =====")
        print("1. Iniciar reconhecimento ao vivo")
        print("2. Exibir histórico de presença")
        print("3. Gerar crachá manualmente")
        print("4. Sair")
        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            recognize_faces_from_camera()
        elif choice == "2":
            show_attendance_history()
        elif choice == "3":
            generate_badge_menu()
        elif choice == "4":
            logging.info("Saindo...")
            break
        else:
            logging.warning("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    setup_database()
    load_known_faces()
    main_menu()
