from ...contents import Image

# from pptx.shapes.shapetree import PicturePlaceholder, SlidePlaceholder
# SlidePlaceholder.insert_picture = PicturePlaceholder.insert_picture
# SlidePlaceholder._new_placeholder_pic = PicturePlaceholder._new_placeholder_pic
# SlidePlaceholder._get_or_add_image = PicturePlaceholder._get_or_add_image


# insert_image = PicturePlaceholder.insert_picture


def replace_with_image(img, shape, slide):
    pic = slide.shapes.add_picture(img, shape.left, shape.top)

    # calculate max width/height for target size
    ratio = min(shape.width / float(pic.width), shape.height / float(pic.height))

    pic.height = int(pic.height * ratio)
    pic.width = int(pic.width * ratio)

    pic.left = int(shape.left + ((shape.width - pic.width) / 2))
    pic.top = int(shape.top + ((shape.height - pic.height) / 2))

    placeholder = shape.element
    placeholder.getparent().remove(placeholder)
    return


class PptImage(Image):
    """Ppt image class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Ppt image class."""
        return cls(parent.image_data)

    # def correct_image(self, placeholder):
    # 	import pdb
    # 	pdb.set_trace()
    # 	width, height = self.image_data.size
    #
    # 	# Make sure the placeholder doesn't zoom in
    # 	placeholder.height = height
    # 	placeholder.width = width
    #
    # 	# Calculate ratios and compare
    # 	image_ratio = width / height
    # 	placeholder_ratio = placeholder.width / placeholder.height
    # 	ratio_difference = placeholder_ratio - image_ratio
    #
    # 	# Placeholder width too wide:
    # 	if ratio_difference > 0:
    # 		difference_on_each_side = ratio_difference / 2
    # 		placeholder.crop_left = -difference_on_each_side
    # 		placeholder.crop_right = -difference_on_each_side
    # 	# Placeholder height too high
    # 	else:
    # 		difference_on_each_side = -ratio_difference / 2
    # 		placeholder.crop_bottom = -difference_on_each_side
    # 		placeholder.crop_top = -difference_on_each_side

    def save(self, placeholder, slide_obj):
        """Saves for Ppt image class."""
        # SlidePlaceholder.insert_picture(placeholder, self.image_data)
        # self.correct_image(placeholder)
        # import pdb
        # pdb.set_trace()
        if self.image_data.__module__.startswith("PIL"):
            import io

            image_data = io.BytesIO()
            self.image_data.save(image_data, format="PNG")
        else:
            image_data = self.image_data
        replace_with_image(image_data, placeholder, slide_obj)
