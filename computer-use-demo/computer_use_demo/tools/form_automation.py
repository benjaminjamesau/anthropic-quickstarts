"""
Advanced form automation system for account creation and form filling.
Supports multiple platforms and intelligent field detection.
"""

import asyncio
from pathlib import Path
import cv2
import numpy as np
import pyautogui
import random
import string
import json
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime

class FormType(Enum):
    EMAIL = "email"
    REDDIT = "reddit"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    GENERIC = "generic"

class FieldType(Enum):
    USERNAME = "username"
    EMAIL = "email"
    PASSWORD = "password"
    CONFIRM_PASSWORD = "confirm_password"
    DATE_OF_BIRTH = "dob"
    PHONE = "phone"
    CAPTCHA = "captcha"

class FormAutomation:
    def __init__(self, computer_tool: Any, captcha_solver: Any):
        self.computer = computer_tool
        self.captcha_solver = captcha_solver
        self.current_form: Optional[Dict[str, Any]] = None
        
        # Load platform-specific form mappings
        self.form_mappings = {
            FormType.EMAIL: {
                "gmail": {
                    "url": "https://accounts.google.com/signup",
                    "fields": {
                        "firstname": {"type": "text", "selector": "input[name='firstName']"},
                        "lastname": {"type": "text", "selector": "input[name='lastName']"},
                        "username": {"type": "text", "selector": "input[name='Username']"},
                        "password": {"type": "password", "selector": "input[name='Passwd']"},
                        "confirm_password": {"type": "password", "selector": "input[name='ConfirmPasswd']"}
                    }
                },
                "outlook": {
                    "url": "https://signup.live.com",
                    "fields": {
                        "email": {"type": "text", "selector": "input[name='MemberName']"},
                        "password": {"type": "password", "selector": "input[name='Password']"}
                    }
                }
            },
            FormType.REDDIT: {
                "url": "https://www.reddit.com/register",
                "fields": {
                    "email": {"type": "text", "selector": "input[id='regEmail']"},
                    "username": {"type": "text", "selector": "input[id='regUsername']"},
                    "password": {"type": "password", "selector": "input[id='regPassword']"}
                }
            }
        }

    async def create_account(self, platform: FormType, **kwargs) -> Dict[str, str]:
        """Create a new account on the specified platform."""
        if platform not in self.form_mappings:
            raise ValueError(f"Unsupported platform: {platform}")

        # Generate random credentials if not provided
        credentials = self._generate_credentials(platform, **kwargs)
        
        # Get platform-specific form mapping
        form_data = self.form_mappings[platform]
        self.current_form = form_data

        # Navigate to signup page
        await self._navigate_to_url(form_data["url"])
        await asyncio.sleep(2)  # Wait for page load

        # Fill out the form
        await self._fill_form(form_data["fields"], credentials)

        # Handle any captchas
        await self._handle_captcha()

        # Submit the form
        await self._submit_form()

        # Save credentials
        self._save_credentials(platform, credentials)

        return credentials

    async def fill_form(self, form_type: FormType, data: Dict[str, str]) -> None:
        """Fill out a form with provided data."""
        # Detect form fields
        fields = await self._detect_form_fields()
        
        # Map detected fields to provided data
        field_mapping = self._map_fields_to_data(fields, data)
        
        # Fill each field
        for field, value in field_mapping.items():
            await self._fill_field(field, value)

    def _generate_credentials(self, platform: FormType, **kwargs) -> Dict[str, str]:
        """Generate random credentials for account creation."""
        credentials = {}
        
        # Generate username if not provided
        if "username" not in kwargs:
            username = self._generate_username()
            credentials["username"] = username
        else:
            credentials["username"] = kwargs["username"]

        # Generate email if not provided
        if "email" not in kwargs:
            email = f"{credentials['username']}_{random.randint(100,999)}@gmail.com"
            credentials["email"] = email
        else:
            credentials["email"] = kwargs["email"]

        # Generate password if not provided
        if "password" not in kwargs:
            password = self._generate_secure_password()
            credentials["password"] = password
            credentials["confirm_password"] = password
        else:
            credentials["password"] = kwargs["password"]
            credentials["confirm_password"] = kwargs["password"]

        # Generate other required fields
        if platform in [FormType.REDDIT, FormType.FACEBOOK]:
            if "dob" not in kwargs:
                credentials["dob"] = self._generate_random_date()

        return credentials

    async def _detect_form_fields(self) -> List[Dict[str, Any]]:
        """Use computer vision to detect form fields on the page."""
        # Take screenshot of the page
        screenshot = await self.computer.screenshot()
        if not screenshot.base64_image:
            raise Exception("Failed to capture screenshot")

        # Convert to CV2 format
        img = self._base64_to_cv2(screenshot.base64_image)
        
        # Detect input fields using template matching
        fields = []
        
        # Common input field types to detect
        field_templates = {
            FieldType.USERNAME: "username_template.png",
            FieldType.EMAIL: "email_template.png",
            FieldType.PASSWORD: "password_template.png",
            FieldType.DATE_OF_BIRTH: "dob_template.png"
        }
        
        for field_type, template_path in field_templates.items():
            matches = self._template_match(img, template_path)
            for match in matches:
                fields.append({
                    "type": field_type,
                    "location": match,
                    "confidence": match[2]
                })
        
        return fields

    async def _fill_field(self, field: Dict[str, Any], value: str) -> None:
        """Fill a single form field."""
        # Move to field location
        x, y = field["location"][:2]
        await self.computer(action="mouse_move", coordinate=(x, y))
        await self.computer(action="left_click")
        
        # Clear existing value
        await self.computer(action="key", text="ctrl+a")
        await self.computer(action="key", text="BackSpace")
        
        # Type new value with human-like delays
        for char in value:
            await self.computer(action="type", text=char)
            await asyncio.sleep(random.uniform(0.1, 0.3))

    async def _handle_captcha(self) -> None:
        """Handle any captchas present on the form."""
        # Detect captcha
        screenshot = await self.computer.screenshot()
        location = await self.captcha_solver.detect_captcha_location(screenshot)
        
        if location:
            # Capture captcha area
            await self.computer(
                action="capture_area",
                coordinate=(
                    location["x"],
                    location["y"],
                    location["width"],
                    location["height"]
                )
            )
            
            # Solve captcha
            solution = await self.captcha_solver.solve_captcha(
                Path("/tmp/captcha.png"),
                captcha_type=location["type"]
            )
            
            # Input solution
            await self.computer(action="type", text=solution)

    def _generate_username(self) -> str:
        """Generate a random username."""
        adjectives = ["happy", "clever", "brave", "bright", "swift"]
        nouns = ["panda", "tiger", "eagle", "wolf", "fox"]
        return f"{random.choice(adjectives)}_{random.choice(nouns)}{random.randint(100,999)}"

    def _generate_secure_password(self) -> str:
        """Generate a secure random password."""
        length = random.randint(12, 16)
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        
        # Ensure password meets common requirements
        password = []
        password.append(random.choice(string.ascii_uppercase))
        password.append(random.choice(string.ascii_lowercase))
        password.append(random.choice(string.digits))
        password.append(random.choice("!@#$%^&*"))
        
        # Fill remaining length with random chars
        for _ in range(length - 4):
            password.append(random.choice(chars))
            
        # Shuffle password
        random.shuffle(password)
        return "".join(password)

    def _generate_random_date(self) -> str:
        """Generate a random date for age verification."""
        year = random.randint(1980, 2000)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{month:02d}/{day:02d}/{year}"

    def _save_credentials(self, platform: FormType, credentials: Dict[str, str]) -> None:
        """Save generated credentials to a secure file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"credentials_{platform.value}_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump({
                "platform": platform.value,
                "timestamp": timestamp,
                "credentials": credentials
            }, f, indent=2)

    @staticmethod
    def _base64_to_cv2(base64_image: str) -> np.ndarray:
        """Convert base64 image to CV2 format."""
        import base64
        import cv2
        import numpy as np
        
        # Decode base64 image
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def _template_match(img: np.ndarray, template_path: str) -> List[Tuple[int, int, float]]:
        """Perform template matching to find UI elements."""
        template = cv2.imread(template_path)
        if template is None:
            return []
            
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        
        locations = np.where(result >= threshold)
        matches = []
        
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], result[pt[1], pt[0]]))
            
        return matches
