from flask import Flask, render_template, request, send_file # type: ignore
import base64, io, os
import numpy as np # type: ignore
from aes_module import derive_key, encrypt_neqr, decrypt_neqr, reconstruct_image
from des_module import derive_des_key, encrypt_des, decrypt_des
from quantum import load_image, pixels_to_neqr
from metrics import mse, mae, psnr

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None
    metrics_data = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "encrypt":
            try:
                aes_input = request.form["aes_key"]
                des_input = request.form["des_key"]

                if not aes_input or not des_input:
                    raise ValueError("Both AES and DES keys must be provided.")

                aes_key = derive_key(aes_input)
                des_key = derive_des_key(des_input)

                file = request.files.get("image")
                if not file:
                    raise ValueError("Please upload an image file.")

                image_path = os.path.join(UPLOAD_FOLDER, "input.png")
                file.save(image_path)

                pixels = load_image(image_path)
                np.save(os.path.join(OUTPUT_FOLDER, "orig_pixels.npy"), pixels)

                neqr_data = pixels_to_neqr(pixels)

                aes_cipher_b64 = encrypt_neqr(
                    neqr_data, 
                    aes_key,
                    os.path.join(OUTPUT_FOLDER, "aes_temp.txt")
                )

                des_cipher_b64 = encrypt_des(aes_cipher_b64, des_key)

                final_path = os.path.join(OUTPUT_FOLDER, "final_encrypted.txt")
                with open(final_path, "w") as f:
                    f.write(des_cipher_b64)

                result = "Image encrypted successfully using AES + DES!"
                return send_file(final_path, as_attachment=True)

            except Exception as e:
                result = f"Encryption failed: {str(e)}"
                return render_template(
                    "index.html",
                    result=result,
                    image_data=None,
                    metrics=None
                )

        elif action == "decrypt":
            try:
                aes_input = request.form["aes_key"]
                des_input = request.form["des_key"]

                if not aes_input or not des_input:
                    raise ValueError("Both AES and DES keys must be provided.")

                aes_key = derive_key(aes_input)
                des_key = derive_des_key(des_input)

                file = request.files.get("ciphertext")
                if not file:
                    raise ValueError("Please upload a ciphertext file.")

                ciphertext_b64 = file.read().decode()

                aes_cipher_b64 = decrypt_des(ciphertext_b64, des_key)

                pixels_list = decrypt_neqr(aes_cipher_b64, aes_key)

                if not pixels_list or len(pixels_list) == 0:
                    raise ValueError("Decryption failed. Incorrect key or corrupt data.")

                dec_pixels_arr = np.array(pixels_list, dtype=np.uint8).reshape((4, 4, 3))

                img = reconstruct_image(pixels_list)
                img_path = os.path.join(OUTPUT_FOLDER, "decrypted.png")
                img.save(img_path)

                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

                try:
                    orig_pixels = np.load(os.path.join(OUTPUT_FOLDER, "orig_pixels.npy"))
                    m = mse(orig_pixels, dec_pixels_arr)
                    a = mae(orig_pixels, dec_pixels_arr)
                    p = psnr(orig_pixels, dec_pixels_arr)

                    metrics_data = {
                        "mse": round(m, 6),
                        "mae": round(a, 6),
                        "psnr_db": float("inf") if p == float("inf") else round(p, 6)
                    }
                except FileNotFoundError:
                    metrics_data = None

                result = "Image decrypted successfully using DES + AES!"
                return render_template(
                    "index.html",
                    result=result,
                    image_data=image_data,
                    metrics=metrics_data
                )

            except (ValueError, UnicodeDecodeError):
                result = "Decryption failed. Incorrect key or corrupted ciphertext."
                return render_template(
                    "index.html",
                    result=result,
                    image_data=None,
                    metrics=None
                )

            except Exception as e:
                result = f"Unexpected error during decryption: {str(e)}"
                return render_template(
                    "index.html",
                    result=result,
                    image_data=None,
                    metrics=None
                )

    return render_template(
        "index.html",
        result=result,
        image_data=image_data,
        metrics=metrics_data
    )


if __name__ == "__main__":
    app.run(debug=True)
