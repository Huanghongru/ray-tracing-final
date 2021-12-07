import taichi as ti
import numpy as np
import argparse, time
from ray_tracing_models import Ray, Camera, Hittable_list, PI, random_in_unit_sphere, refract, reflect, reflectance, random_unit_vector
from instance import Sphere, XYRectangle, XZRectangle, YZRectangle, Box, RotatedY
ti.init(arch=ti.gpu)

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
samples_per_pixel = 4
max_depth = 10
sample_on_unit_sphere_surface = True


@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color += ray_color(ray)
        color /= samples_per_pixel
        canvas[i, j] += color

# Path tracing
@ti.func
def ray_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.8
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        hitrec = scene.hit(Ray(scattered_origin, scattered_direction))
        if hitrec.is_hit:
            if hitrec.material == 0:
                color_buffer = hitrec.color * brightness
                break
            else:
                # Diffuse
                if hitrec.material == 1:
                    target = hitrec.hit_point + hitrec.hit_point_normal
                    if sample_on_unit_sphere_surface:
                        target += random_unit_vector()
                    else:
                        target += random_in_unit_sphere()
                    scattered_direction = target - hitrec.hit_point
                    scattered_origin = hitrec.hit_point
                    brightness *= hitrec.color
                # Metal and Fuzz Metal
                elif hitrec.material == 2 or hitrec.material == 4:
                    fuzz = 0.0
                    if hitrec.material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(),
                                                  hitrec.hit_point_normal)
                    if sample_on_unit_sphere_surface:
                        scattered_direction += fuzz * random_unit_vector()
                    else:
                        scattered_direction += fuzz * random_in_unit_sphere()
                    scattered_origin = hitrec.hit_point
                    if scattered_direction.dot(hitrec.hit_point_normal) < 0:
                        break
                    else:
                        brightness *= hitrec.color
                # Dielectric
                elif hitrec.material == 3:
                    refraction_ratio = 1.5
                    if hitrec.is_front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-scattered_direction.normalized().dot(hitrec.hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hitrec.hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hitrec.hit_point_normal, refraction_ratio)
                    scattered_origin = hitrec.hit_point
                    brightness *= hitrec.color
                brightness /= p_RR
    return color_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Naive Ray Tracing')
    parser.add_argument(
        '--max_depth', type=int, default=20, help='max depth (default: 10)')
    parser.add_argument(
        '--samples_per_pixel', type=int, default=10, help='samples_per_pixel  (default: 4)')
    parser.add_argument(
        '--samples_in_unit_sphere', action='store_true', help='whether sample in a unit sphere')
    args = parser.parse_args()

    max_depth = args.max_depth
    samples_per_pixel = args.samples_per_pixel
    sample_on_unit_sphere_surface = not args.samples_in_unit_sphere
    scene = Hittable_list()

    # Light source
    scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # Ground
    scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # ceiling
    scene.add(Sphere(center=ti.Vector([0, 102.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # back wall
    scene.add(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # right wall
    scene.add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    # left wall
    scene.add(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))

    # Diffuse ball
    # scene.add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])))
    # Metal ball
    # scene.add(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])))
    # Glass ball
    # scene.add(Sphere(center=ti.Vector([0.7, 0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # Metal ball-2
    # scene.add(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.2, material=4, color=ti.Vector([0.8, 0.6, 0.2])))
    # scene.add(Box(low=ti.Vector([0.7, -0.5, -0.5]), high=ti.Vector([0.2, 1.3, -0.2]), material=1, color=ti.Vector([0.6, 0.6, 0.2])))
    # scene.add(XZRectangle(-0.6, 0.6, -0.7, 0.7, 2.5, material=0, color=ti.Vector([20.0, 20.0, 20.0])))
    # scene.add(YZRectangle(-0.6, 0.8, -0.4, 0.4, 1.2, material=0, color=ti.Vector([5.0, 5.0, 3.0])))
    
    box = Box(p0=ti.Vector([0.9, -0.5, -0.5]), p1=ti.Vector([-0.3, 0.6, 0.5]), material=1, color=ti.Vector([0.6, 0.6, 0.2]))
    scene.add(RotatedY(box, 60))

    camera = Camera(aperture=0.0)
    gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
    canvas.fill(0)
    cnt = 0
 
    t1 = time.time()   
    while gui.running:
        render()
        cnt += 1
        gui.set_image(np.sqrt(canvas.to_numpy() / cnt))
        gui.show(f"res/rt-2.png")
    t2 = time.time()
    print(f"complete in {t2-t1:.3f}s")