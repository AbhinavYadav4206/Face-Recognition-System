import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:

    def __init__(self, model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640)):
        self.model_name = model_name
        self.det_size = det_size

        # CPU only
        self.app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])

        self.app.prepare(ctx_id=-1, det_size=self.det_size)

    # Convert image input into OpenCV image
    @staticmethod
    def bytes_to_image(image_data) -> np.ndarray:

        if image_data is None:
            raise ValueError("No image was provided.")

        # Streamlit UploadedFile
        if hasattr(image_data, "getvalue"):
            image_data = image_data.getvalue()

        # File-like object
        elif hasattr(image_data, "read"):
            image_data = image_data.read()

        # Already a NumPy/OpenCV image
        elif isinstance(image_data, np.ndarray):
            return image_data

        # Convert bytes to NumPy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        # Decode image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Unable to read the image.")

        return image


    # Detect faces
    def detect_faces(self, image: np.ndarray):

        if image is None:
            raise ValueError("Image is empty.")

        return self.app.get(image)


    # Get exactly one face
    def _get_single_face(self, image: np.ndarray):

        faces = self.detect_faces(image)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected. "
                "Please provide an image containing exactly one face."
            )

        return faces[0]


    # Normalize embedding
    @staticmethod
    def _normalize_embedding(embedding) -> np.ndarray:

        if embedding is None:
            raise ValueError("Could not generate a face embedding.")

        embedding = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Invalid face embedding.")

        return embedding / norm


    # Generate face embedding
    def get_embedding(self, image_data) -> np.ndarray:

        image = self.bytes_to_image(image_data)

        face = self._get_single_face(image)

        return self._normalize_embedding(face.embedding)


    # Get face information
    def get_face_info(self, image_data) -> dict:

        image = self.bytes_to_image(image_data)

        face = self._get_single_face(image)

        return {
            "embedding": self._normalize_embedding(face.embedding),
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score)
        }


    # Cosine similarity
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:

        embedding1 = np.asarray(embedding1, dtype=np.float32)

        embedding2 = np.asarray(embedding2, dtype=np.float32)

        if embedding1.shape != embedding2.shape:
            raise ValueError("Embedding dimensions do not match.")

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Cannot compare zero embeddings.")

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(np.clip(similarity, -1.0, 1.0))


    # Compare two faces
    @staticmethod
    def compare_faces(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.50) -> tuple[float, bool]:

        similarity = FaceEngine.cosine_similarity(embedding1, embedding2)

        return similarity, similarity >= threshold


# Create reusable face engine
face_engine = FaceEngine()