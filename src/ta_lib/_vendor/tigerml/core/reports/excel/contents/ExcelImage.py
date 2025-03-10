from ...contents import Image
from ..helpers import write_title


class ExcelImage(Image):
    """Excel image class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Excel image class."""
        return cls(parent.image_data)

    # def __new__(cls, input, name='', *args, **kwargs):
    # 	if isinstance(input, Iterable) and not isinstance(input, str):
    # 		from ..Dashboard import ComponentGroup
    # 		return ComponentGroup(input, name=name)

    def save(self, worksheet, workbook, top_row, left_col):
        """Saves for Excel image class."""
        if self.name:
            write_title(self.name, worksheet, workbook, top_row, left_col, self.width)
            top_row += 1
        if self.image_data.__module__.startswith("PIL"):
            import io

            image_data = io.BytesIO()
            self.image_data.save(image_data, format="PNG")
        else:
            image_data = self.image_data
        worksheet.insert_image(top_row, left_col, "", {"image_data": image_data})
