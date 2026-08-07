"""Start the IoU computation server."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gift.inference.iou_server import create_app, parse_args
import uvicorn

if __name__ == "__main__":
    args = parse_args()
    app = create_app(num_workers=args.num_workers)
    uvicorn.run(app, host=args.host, port=args.port)
