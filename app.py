from flask import Flask, render_template, request, send_file 
import base64, io, os
import numpy as np
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

        # ===================== ENCRYPT =====================
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

                # save uploaded image
                image_path = os.path.join(UPLOAD_FOLDER, "input.png")
                file.save(image_path)

                # resize to 4x4 pixels and store them
                pixels = load_image(image_path)  # shape (4,4,3) uint8
                np.save(os.path.join(OUTPUT_FOLDER, "orig_pixels.npy"), pixels)

                # convert pixels to NEQR-style data
                neqr_data = pixels_to_neqr(pixels)

                # AES on NEQR
                aes_cipher_b64 = encrypt_neqr(
                    neqr_data,
                    aes_key,
                    os.path.join(OUTPUT_FOLDER, "aes_temp.txt")
                )

                # DES on AES output
                des_cipher_b64 = encrypt_des(aes_cipher_b64, des_key)

                # write final ciphertext file
                final_path = os.path.join(OUTPUT_FOLDER, "final_encrypted.txt")
                with open(final_path, "w") as f:
                    f.write(des_cipher_b64)

                result = "Image encrypted successfully with AES + DES!"
                return send_file(final_path, as_attachment=True)

            except Exception as e:
                result = f"Encryption failed: {str(e)}"
                return render_template(
                    "index.html",
                    result=result,
                    image_data=None,
                    metrics=None
                )

        # ===================== DECRYPT =====================
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

                # read ciphertext as text
                ciphertext_b64 = file.read().decode()

                # DES -> AES
                aes_cipher_b64 = decrypt_des(ciphertext_b64, des_key)

                # AES -> pixels list
                pixels_list = decrypt_neqr(aes_cipher_b64, aes_key)

                if not pixels_list or len(pixels_list) == 0:
                    raise ValueError("Decryption failed. The provided key may be incorrect.")

                # pixels_list should be length 16 of (r,g,b)
                # reshape to (4,4,3)
                dec_pixels_arr = np.array(pixels_list, dtype=np.uint8).reshape((4, 4, 3))

                # reconstruct PIL image from decrypted pixels
                img = reconstruct_image(pixels_list)
                img_path = os.path.join(OUTPUT_FOLDER, "decrypted.png")
                img.save(img_path)

                # prepare base64 preview to render in template
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # --------- METRICS CALCULATION ---------
                metrics_data = None
                try:
                    # load original reference pixels saved during encryption
                    orig_pixels = np.load(os.path.join(OUTPUT_FOLDER, "orig_pixels.npy"))

                    m = mse(orig_pixels, dec_pixels_arr)
                    a = mae(orig_pixels, dec_pixels_arr)
                    p = psnr(orig_pixels, dec_pixels_arr)

                    # round for nice display
                    metrics_data = {
                        "mse": round(m, 6),
                        "mae": round(a, 6),
                        "psnr_db": (float("inf") if p == float("inf") else round(p, 6))
                    }
                except FileNotFoundError:
                    # no orig reference found (e.g. decrypting on a fresh server)
                    metrics_data = None

                result = "Image decrypted successfully (DES + AES)!"
                return render_template(
                    "index.html",
                    result=result,
                    image_data=image_data,
                    metrics=metrics_data
                )

            except (ValueError, UnicodeDecodeError):
                result = "Decryption failed. The provided key may be incorrect or the file is corrupted."
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

    # GET request
    return render_template(
        "index.html",
        result=result,
        image_data=image_data,
        metrics=metrics_data
    )

if __name__ == "__main__":
    app.run(debug=True)
