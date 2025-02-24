# Ultralytics YOLO 🚀, AGPL-3.0 license

import re

def get_best_youtube_url(url, use_pafy=True):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    This function uses the pafy or yt_dlp library to extract the video info from YouTube. It then finds the highest
    quality MP4 format that has video codec but no audio codec, and returns the URL of this video stream.

    Args:
        url (str): The URL of the YouTube video.
        use_pafy (bool): Use the pafy package, default=True, otherwise use yt_dlp package.

    Returns:
        (str): The URL of the best quality MP4 video stream, or None if no suitable stream is found.
    """
    if use_pafy:
        import pafy  # noqa

        return pafy.new(url).getbestvideo(preftype="mp4").url
    else:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # extract info
        for f in reversed(info_dict.get("formats", [])):  # reversed because best is usually last
            # Find a format with video codec, no audio, *.mp4 extension at least 1920x1080 size
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")


def clean_str(s):
    """
    Cleans a string by replacing special characters with underscore _

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)