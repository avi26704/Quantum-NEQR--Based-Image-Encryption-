from qiskit import QuantumCircuit, transpile  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore
from PIL import Image  # type: ignore
import numpy as np  # type: ignore

path = "sample.png"

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((4, 4)) 
    pixels = np.array(img)
    print("\nOriginal RGB Pixels (4x4):")
    print(pixels)
    return pixels

def pixels_to_neqr(pixels):
    rows, cols, _ = pixels.shape
    representation = []
    for i in range(rows):
        for j in range(cols):
            r, g, b = pixels[i, j]
            pos = format(i * cols + j, '04b')
            r_bin = format(r, '08b')
            g_bin = format(g, '08b')
            b_bin = format(b, '08b')
            representation.append((r_bin, g_bin, b_bin, pos))

    print(len(representation), " -- pixels converted to NEQR format.")
    print("\nRGB NEQR Representation:")
    for r in representation:
        print(r)
    # qc = create_neqr_circuit(representation)
    # simulate_circuit(qc)
    return representation

def create_neqr_circuit(neqr_data):
    qc = QuantumCircuit(28)
    for (r_bin, g_bin, b_bin, pos_bin) in neqr_data:
  
        for i, bit in enumerate(r_bin):
            if bit == '1':
                qc.x(i)

        for i, bit in enumerate(g_bin):
            if bit == '1':
                qc.x(8 + i)

        for i, bit in enumerate(b_bin):
            if bit == '1':
                qc.x(16 + i)

        for i, bit in enumerate(pos_bin):
            if bit == '1':
                qc.x(24 + i)

        qc.barrier()
        qc.reset(range(28))  
    return qc

def simulate_circuit(qc):
    simulator = AerSimulator(method="statevector")
    qc.save_statevector()
    compiled = transpile(qc, simulator, coupling_map=None, optimization_level=0)
    result = simulator.run(compiled).result()
    state = result.data(0)['statevector'].data
    print("\nQuantum Statevector (first 8 shown):")
    for i, amp in enumerate(state[:8]):
        print(f"|{format(i, '028b')}> : {amp}")
    return state
