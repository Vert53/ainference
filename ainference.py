import json
import time
import aiohttp
import asyncio
import aiofiles
from torchvision.datasets import ImageFolder
from aiofiles.threadpool import AsyncBufferedReader
from typing import Tuple

valdir = 'ILSVRC2012/val'
index_to_name_path = 'index_to_name.json'  # mapping  {'791': ['n04204347', 'shopping_cart'],.....}


class ImageNetLoader:
    def __init__(self,
                 folder: ImageFolder):
        self.folder = folder
        self.iter_samples = iter(folder.samples)

    def __aiter__(self):
        return self

    async def __anext__(self) -> Tuple[AsyncBufferedReader, int]:
        try:
            sample_path, target = next(self.iter_samples)
        except StopIteration:
            raise StopAsyncIteration
        async with aiofiles.open(sample_path, 'rb') as sample_file:
            sample = await sample_file.read()
        return sample, target


async def infer_request(session: aiohttp.ClientSession,
                        url: str,
                        sample: AsyncBufferedReader,
                        target: int,
                        queue: asyncio.Queue,
                        ) -> None:
    async with session.post(url, data=sample) as response:
        if response.status == 200:
            output = await response.text()
            await queue.put((output, target))


async def inference_session(url,
                            loader,
                            queue
                            ) -> None:
    async with aiohttp.ClientSession() as session:
        infers = []
        async for sample, target in loader:
            infers.append(asyncio.create_task(infer_request(session=session,
                                                            url=url,
                                                            sample=sample,
                                                            target=target,
                                                            queue=queue)))
        await asyncio.gather(*infers, return_exceptions=True)


class ImagenetPostProcessor:
    def __init__(self):
        self.correct_predictions = 0
        self.total_predictions = 0

    async def postprocess_results(self,
                                  queue: asyncio.Queue,
                                  index_to_name: dict
                                  ) -> None:
        while True:
            output, target = await queue.get()
            top_1_prediction = next(iter(json.loads(output).keys()))
            target_str = index_to_name[str(target)][1]
            if top_1_prediction == target_str:
                self.correct_predictions += 1
            self.total_predictions += 1
            queue.task_done()


async def async_main(url, loader):
    postp = ImagenetPostProcessor()
    with open(index_to_name_path) as f:
        index_to_name = json.load(f)
    queue = asyncio.Queue()
    producer = asyncio.create_task(inference_session(url, loader, queue))
    consumer = asyncio.create_task(postp.postprocess_results(queue, index_to_name),)
    await producer
    await queue.join()
    consumer.cancel()
    print(f'total pred {postp.total_predictions}')
    return postp.correct_predictions/postp.total_predictions


if __name__ == '__main__':
    imagenet_folder = ImageFolder(valdir)
    imagenet_loader = ImageNetLoader(imagenet_folder)
    a = time.time()
    correct = asyncio.run(
        async_main('http://localhost:8080/predictions/resnet50', imagenet_loader)
    )
    b = time.time()
    print(b - a)
    print(correct)
