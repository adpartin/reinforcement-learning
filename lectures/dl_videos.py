import os
from pytube import YouTube


def dl_ytube_video(link, output, overwrite=False):
    """
        Download a youtube video and store it in the data/raw/youtube/video_id folder.
        """
    yt = YouTube(link)
    video = yt.get('mp4', resolution='360p')
    p = '{}.mp4'.format(output)
    if os.path.isfile(p) and not overwrite:
        print('Video already there: {}'.format(p))
    else:
        print('Downloading : {}'.format(link))
        video.download(p,
                       force_overwrite=True)
    return p

vids = [['video1', 'https://www.youtube.com/watch?v=2pWv7GOvuf0'],
        ['video2', 'https://www.youtube.com/watch?v=lfHX2hHRMVQ'],
        ['video3', 'https://www.youtube.com/watch?v=Nd1-UUMVfz4'],
        ['video4', 'https://www.youtube.com/watch?v=PnHCvfgC_ZA'],
        ['video5', 'https://www.youtube.com/watch?v=0g4j2k_Ggc4'],
        ['video6', 'https://www.youtube.com/watch?v=UoPei5o4fps'],
        ['video7', 'https://www.youtube.com/watch?v=KHZVXao4qXs'],
        ['video8', 'https://www.youtube.com/watch?v=ItMutbeOHtc'],
        ['video9', 'https://www.youtube.com/watch?v=sGuiWX07sKw'],
        ['video10', 'https://www.youtube.com/watch?v=kZ_AUmFcZtk']]

for name, link in vids:
    dl_ytube_video(link, name)
