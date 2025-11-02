from flask import Flask, render_template, request, send_file  # type: ignore
import base64, io, os
from aes_module import derive_key, encrypt_neqr, decrypt_neqr, reconstruct_image
from des_module import derive_des_key, encrypt_des, decrypt_des
from quantum import load_image, pixels_to_neqr

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "encrypt":
            try:
                user_key = request.form["key"]
                if not user_key:
                    raise ValueError("Encryption key cannot be empty.")

                aes_key = derive_key(user_key)
                des_key = derive_des_key(user_key)

                file = request.files.get("image")
                if not file:
                    raise ValueError("Please upload an image file.")

                image_path = os.path.join(UPLOAD_FOLDER, "input.png")
                file.save(image_path)

                pixels = load_image(image_path)
                neqr_data = pixels_to_neqr(pixels)

                aes_cipher_b64 = encrypt_neqr(neqr_data, aes_key, os.path.join(OUTPUT_FOLDER, "aes_temp.txt"))
                
                des_cipher_b64 = encrypt_des(aes_cipher_b64, des_key)

                final_path = os.path.join(OUTPUT_FOLDER, "final_encrypted.txt")
                with open(final_path, "w") as f:
                    f.write(des_cipher_b64)

                result = "Image encrypted successfully with AES + DES!"
                return send_file(final_path, as_attachment=True)

            except Exception as e:
                result = f"Encryption failed: {str(e)}"
                return render_template("index.html", result=result, image_data=image_data)

        elif action == "decrypt":
            try:
                user_key = request.form["key"]
                if not user_key:
                    raise ValueError("Decryption key cannot be empty.")

                aes_key = derive_key(user_key)
                des_key = derive_des_key(user_key)

                file = request.files.get("ciphertext")
                if not file:
                    raise ValueError("Please upload a ciphertext file.")

                ciphertext_b64 = file.read().decode()

                aes_cipher_b64 = decrypt_des(ciphertext_b64, des_key)

                pixels = decrypt_neqr(aes_cipher_b64, aes_key)

                if not pixels or len(pixels) == 0:
                    raise ValueError("Decryption failed. The provided key may be incorrect.")

                img = reconstruct_image(pixels)
                img_path = os.path.join(OUTPUT_FOLDER, "decrypted.png")
                img.save(img_path)

                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

                result = "Image decrypted successfully (DES + AES)!"
                return render_template("index.html", result=result, image_data=image_data)

            except (ValueError, UnicodeDecodeError):
                result = "Decryption failed. The provided key may be incorrect or the file is corrupted."
                return render_template("index.html", result=result, image_data=None)

            except Exception as e:
                result = f"Unexpected error during decryption: {str(e)}"
                return render_template("index.html", result=result, image_data=None)

    return render_template("index.html", result=result, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
