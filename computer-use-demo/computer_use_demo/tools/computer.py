import asyncio
import base64
import json
import os
import shlex
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .captcha_solver import CaptchaSolver
from .form_automation import FormAutomation, FormType
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    "solve_captcha",
    "capture_area",
    "create_account",
    "fill_form",
]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 0.5
    _scaling_enabled = False

    @property
    def options(self) -> ComputerToolOptions:
        return {
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()
        
        # Remove resolution restrictions
        self.width = int(os.getenv("WIDTH") or 1920)  # Default to 1920x1080
        self.height = int(os.getenv("HEIGHT") or 1080)
        
        # Initialize tools
        self.captcha_solver = CaptchaSolver(api_key=os.getenv("CAPTCHA_API_KEY"))
        self.form_automation = FormAutomation(self, self.captcha_solver)
        self.capture_area = None
        
        # Set up display
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
            self._display_prefix = f"DISPLAY=:{self.display_num} "
        else:
            self.display_num = None
            self._display_prefix = "DISPLAY=:0 "  # Default to primary display
            
        self.xdotool = f"{self._display_prefix}xdotool"

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        # Remove coordinate validation
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            
            x, y = coordinate[0], coordinate[1]
            
            if action == "mouse_move":
                return await self.shell(f"{self.xdotool} mousemove --sync {x} {y}")
            elif action == "left_click_drag":
                return await self.shell(
                    f"{self.xdotool} mousedown 1 mousemove --sync {x} {y} mouseup 1"
                )

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                return await self.shell(f"{self.xdotool} key -- {text}")
            elif action == "type":
                results: list[ToolResult] = []
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    cmd = f"{self.xdotool} type --delay {TYPING_DELAY_MS} -- {shlex.quote(chunk)}"
                    results.append(await self.shell(cmd, take_screenshot=False))
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(
                    output="".join(result.output or "" for result in results),
                    error="".join(result.error or "" for result in results),
                    base64_image=screenshot_base64,
                )

        if action == "solve_captcha":
            # Handle captcha solving
            screenshot = await self.screenshot()
            if not screenshot.base64_image:
                raise ToolError("Failed to capture captcha screenshot")

            # Save screenshot to temporary file
            temp_path = Path(OUTPUT_DIR) / f"captcha_{uuid4().hex}.png"
            temp_path.write_bytes(base64.b64decode(screenshot.base64_image))

            try:
                solution = await self.captcha_solver.solve_image_captcha(temp_path)
                return ToolResult(
                    output=f"Captcha solution: {solution}", base64_image=screenshot.base64_image
                )
            finally:
                temp_path.unlink(missing_ok=True)

        elif action == "capture_area":
            if not coordinate or len(coordinate) != 4:
                raise ToolError("capture_area requires x, y, width, height coordinates")
            self.capture_area = ",".join(map(str, coordinate))
            return await self.screenshot()

        elif action == "create_account":
            if "platform" not in kwargs:
                raise ToolError("platform is required for account creation")
            try:
                platform = FormType(kwargs["platform"])
                credentials = await self.form_automation.create_account(
                    platform,
                    **{k: v for k, v in kwargs.items() if k != "platform"}
                )
                return ToolResult(
                    output=f"Account created successfully: {json.dumps(credentials, indent=2)}"
                )
            except Exception as e:
                raise ToolError(f"Failed to create account: {str(e)}")

        elif action == "fill_form":
            if "form_type" not in kwargs or "data" not in kwargs:
                raise ToolError("form_type and data are required for form filling")
            try:
                form_type = FormType(kwargs["form_type"])
                await self.form_automation.fill_form(form_type, kwargs["data"])
                return ToolResult(output="Form filled successfully")
            except Exception as e:
                raise ToolError(f"Failed to fill form: {str(e)}")

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                result = await self.shell(
                    f"{self.xdotool} getmouselocation --shell",
                    take_screenshot=False,
                )
                output = result.output or ""
                x, y = int(output.split("X=")[1].split("\n")[0]), int(output.split("Y=")[1].split("\n")[0])
                return result.replace(output=f"X={x},Y={y}")
            else:
                click_arg = {
                    "left_click": "1",
                    "right_click": "3",
                    "middle_click": "2",
                    "double_click": "--repeat 2 --delay 500 1",
                }[action]
                return await self.shell(f"{self.xdotool} click {click_arg}")

        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen or specific area."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        if self.capture_area:
            # Take screenshot of specific area
            screenshot_cmd = f"{self._display_prefix}scrot -a {self.capture_area} {path}"
            self.capture_area = None  # Reset after use
        else:
            # Take full screenshot
            if shutil.which("gnome-screenshot"):
                screenshot_cmd = f"{self._display_prefix}gnome-screenshot -f {path} -p"
            else:
                screenshot_cmd = f"{self._display_prefix}scrot {path}"

        result = await self.shell(screenshot_cmd, take_screenshot=False)
        if path.exists():
            return result.replace(
                base64_image=base64.b64encode(path.read_bytes()).decode()
            )
        raise ToolError(f"Failed to take screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        return x, y
