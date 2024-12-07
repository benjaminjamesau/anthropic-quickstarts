"""
Advanced captcha solving capabilities with multiple service providers and ML-based detection.
"""

import base64
from pathlib import Path
import asyncio
import httpx
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

class CaptchaType(Enum):
    IMAGE = "image"
    RECAPTCHA = "recaptcha"
    HCAPTCHA = "hcaptcha"
    FUNCAPTCHA = "funcaptcha"
    TEXT = "text"

class CaptchaService(Enum):
    TWOCAPTCHA = "2captcha"
    ANTICAPTCHA = "anticaptcha"
    CAPSOLVER = "capsolver"
    DEATHBYCAPTCHA = "deathbycaptcha"
    CAPMONSTER = "capmonster"

class CaptchaSolver:
    def __init__(self, api_key: str = None, service: str = "2captcha"):
        self.api_key = api_key
        self.service = CaptchaService(service)
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Service API endpoints
        self._endpoints = {
            CaptchaService.TWOCAPTCHA: "https://2captcha.com/in.php",
            CaptchaService.ANTICAPTCHA: "https://api.anti-captcha.com",
            CaptchaService.CAPSOLVER: "https://api.capsolver.com",
            CaptchaService.DEATHBYCAPTCHA: "https://api.deathbycaptcha.com/2.0",
            CaptchaService.CAPMONSTER: "https://api.capmonster.cloud"
        }

    async def solve_captcha(self, image_path: Path, captcha_type: CaptchaType = CaptchaType.IMAGE) -> str:
        """Main method to solve any type of captcha using the configured service."""
        # First detect the type if not specified
        if captcha_type == CaptchaType.IMAGE:
            detected_type = await self.detect_captcha_type(image_path)
            captcha_type = detected_type or captcha_type

        # Route to appropriate solver
        if captcha_type == CaptchaType.IMAGE:
            return await self._solve_image_captcha(image_path)
        elif captcha_type == CaptchaType.RECAPTCHA:
            return await self._solve_recaptcha(image_path)
        elif captcha_type == CaptchaType.HCAPTCHA:
            return await self._solve_hcaptcha(image_path)
        elif captcha_type == CaptchaType.FUNCAPTCHA:
            return await self._solve_funcaptcha(image_path)
        else:
            raise ValueError(f"Unsupported captcha type: {captcha_type}")

    async def _solve_image_captcha(self, image_path: Path) -> str:
        """Solve image-based captcha using the configured service."""
        solver_methods = {
            CaptchaService.TWOCAPTCHA: self._solve_with_2captcha,
            CaptchaService.ANTICAPTCHA: self._solve_with_anticaptcha,
            CaptchaService.CAPSOLVER: self._solve_with_capsolver,
            CaptchaService.DEATHBYCAPTCHA: self._solve_with_deathbycaptcha,
            CaptchaService.CAPMONSTER: self._solve_with_capmonster
        }
        
        solver = solver_methods.get(self.service)
        if not solver:
            raise ValueError(f"Unsupported service: {self.service}")
        
        return await solver(image_path)

    async def detect_captcha_type(self, image_path: Path) -> Optional[CaptchaType]:
        """Use ML and image processing to detect captcha type."""
        # Convert image to CV2 format
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Check for reCAPTCHA logo
        if self._detect_recaptcha_elements(img):
            return CaptchaType.RECAPTCHA
            
        # 2. Check for hCaptcha elements
        if self._detect_hcaptcha_elements(img):
            return CaptchaType.HCAPTCHA
            
        # 3. Check for FunCaptcha elements
        if self._detect_funcaptcha_elements(img):
            return CaptchaType.FUNCAPTCHA
            
        # 4. If none of the above, assume it's a standard image captcha
        return CaptchaType.IMAGE

    def _detect_recaptcha_elements(self, img: np.ndarray) -> bool:
        """Detect reCAPTCHA-specific elements in the image."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color range for Google's blue
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask and find contours
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular shapes typical of reCAPTCHA
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 0.9 <= w/h <= 1.1:  # Square aspect ratio
                return True
        return False

    async def _solve_with_capsolver(self, image_path: Path) -> str:
        """Solve captcha using Capsolver service."""
        if not self.api_key:
            raise ValueError("Capsolver API key is required")

        # Read and encode image
        image_data = base64.b64encode(image_path.read_bytes()).decode()
        
        # Prepare request
        payload = {
            "clientKey": self.api_key,
            "task": {
                "type": "ImageToTextTask",
                "body": image_data,
            }
        }
        
        async with self.client as client:
            # Create task
            response = await client.post(
                f"{self._endpoints[CaptchaService.CAPSOLVER]}/createTask",
                json=payload
            )
            result = response.json()
            
            if "errorId" in result and result["errorId"] > 0:
                raise Exception(f"Capsolver error: {result.get('errorDescription')}")
                
            task_id = result["taskId"]
            
            # Get solution
            for _ in range(30):
                await asyncio.sleep(5)
                status_payload = {
                    "clientKey": self.api_key,
                    "taskId": task_id
                }
                response = await client.post(
                    f"{self._endpoints[CaptchaService.CAPSOLVER]}/getTaskResult",
                    json=status_payload
                )
                result = response.json()
                
                if result.get("status") == "ready":
                    return result["solution"]["text"]
                    
            raise TimeoutError("Capsolver solving timed out")

    async def _solve_with_deathbycaptcha(self, image_path: Path) -> str:
        """Solve captcha using DeathByCaptcha service."""
        if not self.api_key:
            raise ValueError("DeathByCaptcha API key is required")

        # Read and encode image
        image_data = base64.b64encode(image_path.read_bytes()).decode()
        
        # Prepare request
        payload = {
            "username": self.api_key.split(':')[0],
            "password": self.api_key.split(':')[1],
            "captcha": image_data
        }
        
        async with self.client as client:
            # Submit captcha
            response = await client.post(
                f"{self._endpoints[CaptchaService.DEATHBYCAPTCHA]}/captcha",
                json=payload
            )
            result = response.json()
            
            if "status" not in result or result["status"] != 0:
                raise Exception(f"DeathByCaptcha error: {result.get('error')}")
                
            captcha_id = result["captcha"]
            
            # Get solution
            for _ in range(30):
                await asyncio.sleep(5)
                response = await client.get(
                    f"{self._endpoints[CaptchaService.DEATHBYCAPTCHA]}/captcha/{captcha_id}"
                )
                result = response.json()
                
                if result.get("text"):
                    return result["text"]
                    
            raise TimeoutError("DeathByCaptcha solving timed out")

    async def _solve_with_capmonster(self, image_path: Path) -> str:
        """Solve captcha using CapMonster service."""
        if not self.api_key:
            raise ValueError("CapMonster API key is required")

        # Read and encode image
        image_data = base64.b64encode(image_path.read_bytes()).decode()
        
        # Prepare request
        payload = {
            "clientKey": self.api_key,
            "task": {
                "type": "ImageToTextTask",
                "body": image_data,
            }
        }
        
        async with self.client as client:
            # Create task
            response = await client.post(
                f"{self._endpoints[CaptchaService.CAPMONSTER]}/createTask",
                json=payload
            )
            result = response.json()
            
            if result.get("errorId") > 0:
                raise Exception(f"CapMonster error: {result.get('errorDescription')}")
                
            task_id = result["taskId"]
            
            # Get solution
            for _ in range(30):
                await asyncio.sleep(5)
                status_payload = {
                    "clientKey": self.api_key,
                    "taskId": task_id
                }
                response = await client.post(
                    f"{self._endpoints[CaptchaService.CAPMONSTER]}/getTaskResult",
                    json=status_payload
                )
                result = response.json()
                
                if result.get("status") == "ready":
                    return result["solution"]["text"]
                    
            raise TimeoutError("CapMonster solving timed out")

    async def detect_captcha_location(self, screenshot_path: Path) -> Optional[Dict[str, Any]]:
        """
        Advanced captcha detection using computer vision.
        Returns location and type of detected captcha.
        """
        img = cv2.imread(str(screenshot_path))
        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Template matching for common captcha providers
        captcha_locations = []
        
        # Check for reCAPTCHA
        if self._detect_recaptcha_elements(img):
            captcha_locations.append({
                "type": CaptchaType.RECAPTCHA,
                "confidence": 0.95
            })
            
        # Check for text-based captchas using OCR
        text_regions = self._detect_text_regions(gray)
        if text_regions:
            captcha_locations.extend([
                {
                    "type": CaptchaType.TEXT,
                    "confidence": 0.8,
                    "bbox": region
                }
                for region in text_regions
            ])
            
        # Return the most confident detection
        if captcha_locations:
            best_match = max(captcha_locations, key=lambda x: x["confidence"])
            if "bbox" in best_match:
                x, y, w, h = best_match["bbox"]
                return {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "type": best_match["type"].value,
                    "confidence": best_match["confidence"]
                }
            
        return None

    def _detect_text_regions(self, gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions that might be captchas."""
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter based on aspect ratio and size
            if 0.2 <= h/w <= 0.8 and w > 30 and h > 10:
                # Use OCR to verify text presence
                roi = gray_img[y:y+h, x:x+w]
                text = pytesseract.image_to_string(roi)
                if text.strip():
                    text_regions.append((x, y, w, h))
                    
        return text_regions
