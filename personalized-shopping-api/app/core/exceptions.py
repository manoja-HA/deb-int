"""
Custom exceptions and exception handlers
"""

from typing import Any, Dict
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

# ===== Custom Exceptions =====

class AppException(Exception):
    """Base application exception"""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Dict[str, Any] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class NotFoundException(AppException):
    """Resource not found exception"""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} not found: {identifier}",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource": resource, "identifier": identifier},
        )

class ValidationException(AppException):
    """Validation error exception"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details or {},
        )

class WorkflowException(AppException):
    """Multi-agent workflow execution exception"""

    def __init__(self, message: str, agent: str = None):
        super().__init__(
            message=f"Workflow error: {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"agent": agent} if agent else {},
        )

class CacheException(AppException):
    """Cache operation exception"""

    def __init__(self, message: str):
        super().__init__(
            message=f"Cache error: {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# ===== Exception Handlers =====

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """
    Handle custom application exceptions

    Returns consistent error response format
    """
    logger.error(
        f"Application exception: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "status_code": exc.status_code,
                "details": exc.details,
                "path": str(request.url.path),
            }
        },
    )

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle Pydantic validation errors

    Returns detailed validation error messages
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(
        "Validation error",
        extra={
            "errors": errors,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "details": {"validation_errors": errors},
                "path": str(request.url.path),
            }
        },
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions

    Logs full stack trace and returns generic error message
    """
    logger.error(
        f"Unexpected exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path},
    )

    # Don't expose internal errors in production
    message = str(exc) if not settings.ENVIRONMENT == "production" else "Internal server error"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": message,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "path": str(request.url.path),
            }
        },
    )

from app.core.config import settings
