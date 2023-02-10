import argparse
from siren.data_loader import get_data_loader_cls_by_type
from siren.optimizer import JaxOptimizer
from siren.model import get_model_cls_by_type
from util.log import Logger
from util.timer import Timer

import optax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints
from siren.network_flax import Siren
import orbax
import orbax.checkpoint as orbax


def parse_args():
    parser = argparse.ArgumentParser(description="Train SirenHighres")

    parser.add_argument("--file", type=str, help="location of the file", required=True)
    parser.add_argument(
        "--nc",
        type=int,
        default=3,
        help="number of channels of input image. if the source is color (3) and --nc is 1, then the source is converted to gray scale",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="normal",
        choices=["normal", "gradient", "laplacian", "combined"],
        help="training image type",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="resize the image to this (squre) shape. 0 if not goint go resize",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="the size of batches. 0 for single batch",
    )
    parser.add_argument("--epoch", type=int, default=10000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--print_iter", type=int, default=200, help="when to print intermediate info"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="256,256,256",
        help="layers of multi layer perceptron",
    )
    parser.add_argument("--omega", type=float, default=30, help="omega value of Siren")

    args = parser.parse_args()
    return args


def main(args):
    layers = [int(l) for l in args.layers.split(",")]

    Model = get_model_cls_by_type(args.type)
    DataLoader = get_data_loader_cls_by_type(args.type)

    data_loader = DataLoader(args.file, args.nc, args.size, args.batch_size)
    # model = Model(layers, args.nc, args.omega)
    # optimizer = JaxOptimizer("adam", model, args.lr)

    key = jax.random.PRNGKey(0)
    key, network_key = jax.random.split(key)
    model = Siren(num_channels=layers, output_dim=1)
    state = TrainState.create(apply_fn=model.apply, 
                              params=model.init(network_key, jnp.ones((args.batch_size, 2))),
                              tx=optax.adam(learning_rate=args.lr))

    @jax.jit
    def update(state, x, y):
        def jacobian(apply_fn, params, x):
            f = lambda x: apply_fn(params, x)
            y, f_vjp = jax.vjp(f, x)
            (x_grad,) = f_vjp(jnp.ones_like(y))
            return x_grad
            
        def mse_loss(params):
            output = jacobian(state.apply_fn, params, x)
            diff = output - y
            return jnp.mean(jnp.sum(diff**2, axis=-1))
        # def mse_loss(params):
        #     output = state.apply_fn(params, x)
        #     return jnp.mean((output - y)**2)
        loss, grads = jax.value_and_grad(mse_loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
        
    name = args.file.split(".")[0]
    logger = Logger(name)
    logger.save_option(vars(args))

    gt_img = data_loader.get_ground_truth_image()
    logger.save_image("original", data_loader.original_pil_img)
    logger.save_image("gt", gt_img)

    iter_timer = Timer()
    iter_timer.start()

    def interm_callback(i, data, params):
        log = {}
        # loss = model.loss_func(params, data)
        log["loss"] = float(loss)
        log["iter"] = i
        log["duration_per_iter"] = iter_timer.get_dt() / args.print_iter

        logger.save_log(log)
        print(log)

    def interm_callback_2(i, loss):
        log = {}
        # loss = model.loss_func(params, data)
        log["loss"] = float(loss)
        log["iter"] = i
        log["duration_per_iter"] = iter_timer.get_dt() / args.print_iter

        logger.save_log(log)
        print(log)

    print("Training Start")
    print(vars(args))

    total_timer = Timer()
    total_timer.start()
    last_data = None
    for e in range(args.epoch):
        data_loader = DataLoader(args.file, args.nc, args.size, args.batch_size)
        total_loss = []
        for data in data_loader:
              state, loss = update(state, data['input'], data['output'])
              total_loss.append(loss.item())
        if e % args.print_iter == 0:
            ckpt = {'model': state}
            orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
            checkpoints.save_checkpoint(ckpt_dir='CKPT_jacob',
                                        target=ckpt,
                                        step=e,
                                        overwrite=False,
                                        keep=2,
                                        orbax_checkpointer=orbax_checkpointer)


            interm_callback_2(e, jnp.mean(jnp.array(total_loss)))
        # print(f"epoch {e} loss: {jnp.mean(jnp.array(total_loss))}")
    #         optimizer.step(data)
    #         last_data = data
    #         if optimizer.iter_cnt % args.print_iter == 0:
    #             interm_callback(
    #                 optimizer.iter_cnt, data, optimizer.get_optimized_params()
    #             )

    # if not optimizer.iter_cnt % args.print_iter == 0:
    #     interm_callback(optimizer.iter_cnt, data, optimizer.get_optimized_params())

    train_duration = total_timer.get_dt()
    print("Training Duration: {} sec".format(train_duration))
    # logger.save_net_params(optimizer.get_optimized_params())
    logger.save_losses_plot()


if __name__ == "__main__":
    args = parse_args()
    main(args)
