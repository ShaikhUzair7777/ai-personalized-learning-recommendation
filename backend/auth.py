from fastapi import Header, HTTPException
from backend.supabase_client import supabase


async def get_authenticated_user(
    authorization: str | None = Header(default=None),
):
    """
    Validate the Supabase access token and return the authenticated user.

    Expected header:
        Authorization: Bearer <access_token>
    """

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is required.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header must use Bearer token.",
        )

    access_token = authorization.split(" ", 1)[1].strip()

    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Access token is missing.",
        )

    try:
        response = supabase.auth.get_user(access_token)

        user = response.user

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired access token.",
            )

        return user

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Authentication error: {exc}")

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token.",
        )