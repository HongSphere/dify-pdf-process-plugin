import io
from collections.abc import Generator
from typing import Any, Optional

import pymupdf
from PIL import Image
from dify_plugin.entities import I18nObject
from dify_plugin.entities.tool import ToolInvokeMessage, ToolParameter
from dify_plugin import Tool
from dify_plugin.file.file import File


class PDFToPNGTool(Tool):
    """
    A tool for converting PDF files to PNG images.
    This tool takes a PDF file as input and returns each page as a separate PNG image.
    """

    def _invoke(
        self,
        tool_parameters: dict[str, Any],
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Convert a PDF file to PNG images.

        Args:
            tool_parameters (dict[str, Any]): Parameters for the tool
                - pdf_content (File): Dify File object containing the PDF
                - zoom (float): Zoom factor for image quality (default is 2)
                - max_image_size (Optional[float]): Maximum image size limit in MB (optional)
            user_id (Optional[str]): The ID of the user invoking the tool
            conversation_id (Optional[str]): The conversation ID
            app_id (Optional[str]): The app ID
            message_id (Optional[str]): The message ID

        Returns:
            Generator[ToolInvokeMessage, None, None]: Generator yielding the PNG images

        Raises:
            ValueError: If the PDF content format is invalid or required parameters are missing
            Exception: For any other errors during PDF processing
        """
        doc = None
        try:
            # Get and validate parameters
            pdf_content = tool_parameters.get("pdf_content")
            if not isinstance(pdf_content, File):
                raise ValueError("Invalid PDF content format. Expected File object.")

            # Get zoom parameter with default value
            zoom_param = tool_parameters.get("zoom")
            try:
                zoom = 2.0 if zoom_param is None else float(zoom_param)
                if zoom <= 0:
                    zoom = 2.0
            except (ValueError, TypeError):
                zoom = 2.0

            # Get max_image_size parameter (in MB)
            max_image_size_param = tool_parameters.get("max_image_size")
            max_size_bytes: Optional[int] = None
            if (
                max_image_size_param is not None
                and str(max_image_size_param).strip() != ""
            ):
                try:
                    max_size_val = float(max_image_size_param)
                    if max_size_val > 0:
                        max_size_bytes = int(max_size_val * 1024 * 1024)
                except (ValueError, TypeError):
                    max_size_bytes = None

            original_filename = pdf_content.filename or "document"
            base_filename = original_filename.rsplit(".", 1)[0]

            # Convert bytes to BytesIO object
            pdf_bytes_io = io.BytesIO(pdf_content.blob)

            try:
                # Open PDF with PyMuPDF
                doc = pymupdf.open(stream=pdf_bytes_io, filetype="pdf")
            except Exception as e:
                raise ValueError(f"Invalid PDF file: {str(e)}")

            total_pages = doc.page_count
            if total_pages == 0:
                raise ValueError("The PDF file contains no pages.")

            # Process each page
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                current_zoom = zoom
                min_zoom = 0.1
                max_attempts = 15
                attempt = 0

                # If a size limit is configured, perform a zero-overhead physical pre-check:
                # Even under an extremely optimistic 1:6 PNG compression ratio,
                # each byte of compressed file corresponds to at most ~2 raw RGB pixels.
                # If initial pixels exceed max_size_bytes * 2, it is physically impossible
                # to fit within max_image_size. Pre-clip initial zoom to avoid rendering
                # a guaranteed-to-fail giant image (e.g. 100M+ pixels) that takes 30s+ to compress.
                if max_size_bytes is not None:
                    page_area = page.rect.width * page.rect.height
                    if page_area > 0:
                        max_possible_pixels = max_size_bytes * 2
                        current_pixels = page_area * (current_zoom ** 2)
                        if current_pixels > max_possible_pixels:
                            pre_clip_zoom = round((max_possible_pixels / page_area) ** 0.5, 2)
                            current_zoom = max(min_zoom, min(current_zoom, pre_clip_zoom))

                final_blob = b""
                while True:
                    mat = pymupdf.Matrix(current_zoom, current_zoom)
                    pix = page.get_pixmap(matrix=mat)

                    # Convert PyMuPDF pixmap to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # Release pixmap memory
                    pix = None

                    # Save PIL Image to an in-memory bytes buffer
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="PNG")
                    img_bytes = img_buffer.getvalue()
                    img_size = len(img_bytes)
                    final_blob = img_bytes

                    # Check if size constraint is satisfied
                    if max_size_bytes is None or img_size <= max_size_bytes:
                        break

                    attempt += 1
                    if current_zoom <= min_zoom or attempt >= max_attempts:
                        break

                    # Estimate target zoom factor directly to reach size constraint in one step
                    # Using area scaling ratio with a 0.88 safety margin for PNG non-linear compression
                    ratio = (max_size_bytes / img_size) ** 0.5
                    scale_step = max(0.1, ratio * 0.88)
                    next_zoom = round(current_zoom * scale_step, 2)
                    if next_zoom >= current_zoom:
                        next_zoom = round(current_zoom - 0.1, 2)
                    current_zoom = max(min_zoom, next_zoom)

                # Create filename for this page
                output_filename = f"{base_filename}_page{page_num + 1}.png"

                # Send the PNG image
                yield self.create_blob_message(
                    blob=final_blob,
                    meta={"mime_type": "image/png", "file_name": output_filename},
                )

            # Send completion message
            yield self.create_text_message(
                f"Successfully converted {total_pages} pages to PNG images."
            )

        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Error converting PDF to PNG: {str(e)}")
        finally:
            if doc:
                doc.close()

    def get_runtime_parameters(
        self,
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> list[ToolParameter]:
        """
        Get the runtime parameters for the PDF to PNG conversion tool.

        Returns:
            list[ToolParameter]: List of tool parameters
        """
        parameters = [
            ToolParameter(
                name="pdf_content",
                label=I18nObject(en_US="PDF Content", zh_Hans="PDF 内容"),
                human_description=I18nObject(
                    en_US="PDF file to convert to PNG images",
                    zh_Hans="要转换为PNG图片的PDF文件",
                ),
                type=ToolParameter.ToolParameterType.FILE,
                form=ToolParameter.ToolParameterForm.FORM,
                required=True,
                file_accepts=["application/pdf"],
            ),
            ToolParameter(
                name="zoom",
                label=I18nObject(en_US="Zoom Factor", zh_Hans="缩放因子"),
                human_description=I18nObject(
                    en_US="Zoom factor for image quality (default is 2)",
                    zh_Hans="图像质量的缩放因子（默认为2）",
                ),
                type=ToolParameter.ToolParameterType.NUMBER,
                form=ToolParameter.ToolParameterForm.FORM,
                required=False,
                default=2,
            ),
            ToolParameter(
                name="max_image_size",
                label=I18nObject(
                    en_US="Max Image Size (MB)", zh_Hans="图片大小限制 (MB)"
                ),
                human_description=I18nObject(
                    en_US="Max size in MB (auto-downgrades if exceeded, optional)",
                    zh_Hans="单张图片大小上限（MB），超限自动降级，留空不限制",
                ),
                type=ToolParameter.ToolParameterType.NUMBER,
                form=ToolParameter.ToolParameterForm.FORM,
                required=False,
            ),
        ]
        return parameters
