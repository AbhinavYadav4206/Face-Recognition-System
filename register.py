from database import add_person, get_all_embeddings
from face_engine import face_engine

# Starting threshold for duplicate detection
DUPLICATE_THRESHOLD = 0.60


def register_face(name: str, image_file) -> dict:

    # Validate input
    if not name or not name.strip():
        return {
            "success": False,
            "message": "Please enter a name.",
            "person_id": None,
            "duplicate": False
        }

    if image_file is None:
        return {
            "success": False,
            "message": "Please provide an image.",
            "person_id": None,
            "duplicate": False
        }

    name = name.strip()

    # Generate face embedding
    try:
        embedding = face_engine.get_embedding(image_file)

    except ValueError as error:
        return {
            "success": False,
            "message": str(error),
            "person_id": None,
            "duplicate": False
        }

    except Exception as error:
        return {
            "success": False,
            "message": f"Face processing failed: {error}",
            "person_id": None,
            "duplicate": False
        }

    # Get existing registered faces
    try:
        registered_people = get_all_embeddings()

    except Exception as error:
        return {
            "success": False,
            "message": f"Database error: {error}",
            "person_id": None,
            "duplicate": False
        }

    # Check for duplicate
    for person in registered_people:

        try:
            similarity = face_engine.cosine_similarity(
                embedding,
                person["embedding"]
            )

        except ValueError:
            continue

        # Duplicate found
        if similarity >= DUPLICATE_THRESHOLD:

            return {
                "success": False,
                "message": (
                    f"Person already registered as "
                    f"{person['name']}."
                ),
                "person_id": person["person_id"],
                "duplicate": True,
                "similarity": similarity
            }

    # No duplicate -> add to database
    try:
        result = add_person(
            name=name,
            embedding=embedding
        )

    except Exception as error:

        return {
            "success": False,
            "message": f"Database error: {error}",
            "person_id": None,
            "duplicate": False
        }

    # Success
    return {
        "success": True,
        "message": "Face registered successfully.",
        "person_id": result["person_id"],
        "duplicate": False
    }